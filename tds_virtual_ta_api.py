import json
from openai import OpenAI
import os
import re
from collections import defaultdict
from typing import List
import uvicorn
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
from dotenv import load_dotenv
from typing import Optional, List
import os
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Document structure for storage and indexing
class Document:
    def __init__(self, content, url, source_type, metadata=None):
        self.content = content
        self.url = url
        self.source_type = source_type
        self.metadata = metadata or {}

# FastAPI request model
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # assuming base64 if ever used

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]
    

# Core RAG pipeline
class RagPipeline:
    def __init__(self, course_file: str, discourse_file: str):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_data(course_file, discourse_file)
        self.build_index()


    def get_followup_ta_post(self, base_doc: Document) -> Optional[Document]:
        topic_id = base_doc.metadata.get("topic_id")
        base_post_number = self.extract_post_number(base_doc.url)
        ta_usernames = {'Jivraj', 'Suchintika', 'Hritik Roshan', 'carlton', 'Prasanna', 's.anand'}

        # Find all posts in the same topic with post_number > current
        candidates = [
            doc for doc in self.documents
            if doc.source_type == 'discourse'
            and doc.metadata.get("topic_id") == topic_id
            and self.extract_post_number(doc.url) > base_post_number
            and doc.metadata.get("username") in ta_usernames
        ]

        if candidates:
            # Return the earliest TA follow-up (lowest post number)
            return sorted(candidates, key=lambda d: self.extract_post_number(d.url))[0]
        return None



    def reconstruct_discourse_url(self, original_url: str, topic_id: str) -> str:
        """
        Convert URL from:
            /t/<slug-with-topicid>/<post_number>
        or
            /t/<slug>/<post_number>
        
        To:
            https://discourse.onlinedegree.iitm.ac.in/t/<slug-fixed>/<topic_id>/<post_number>

        Adds '-2025' to slug if it contains one of ['jan', 'feb', 'mar', 'apr'] but not '2025'.
        """
        base_url = "https://discourse.onlinedegree.iitm.ac.in"

        # Extract slug and post_number from original URL
        m = re.search(r'/t/([^/]+)/(\d+)$', original_url)
        if not m:
            return original_url  # fallback

        slug_with_id = m.group(1)
        # post_number = m.group(2)  # now passed explicitly

        # Remove trailing topic_id from slug if present
        parts = slug_with_id.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            slug = parts[0]
        else:
            slug = slug_with_id

        # Add "-2025" to slug if it contains month keywords and not already has 2025
        slug_lower = slug.lower()
        if any(month in slug_lower for month in ['jan', 'feb', 'mar', 'apr']) and '2025' not in slug:
            slug += '-2025'

        # Construct final URL
        new_url = f"{base_url}/t/{slug}/{topic_id}"
        return new_url



    def extract_post_number(self, url):
        match = re.search(r'/(\d+)$', url)
        return int(match.group(1)) if match else 0

    def merge_adjacent_posts(self, posts):
        merged_docs = []
        i = 0
        while i < len(posts):
            current = posts[i]
            merged_content = f"[{current['username']}] {current['content'].strip()}"
            metadata = {
                'topic_id': current['topic_id'],
                'url': current['url'],
                'username': current['username']
            }
            i += 1
            while i < len(posts) and posts[i]['username'] in {'Jivraj', 'Suchintika', 'Hritik Roshan', 'carlton', 'Prasanna', 's.anand'}:
                merged_content += f"\n[{posts[i]['username']}] {posts[i]['content'].strip()}"
                i += 1
            merged_docs.append(Document(
                content=merged_content,
                url=metadata['url'],
                source_type='discourse',
                metadata=metadata
            ))
        return merged_docs

    def load_data(self, course_file: str, discourse_file: str):
        # Load course content
        with open(course_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.documents.append(Document(
                    content=data['content'],
                    url=data['url'],
                    source_type='course'
                ))

        # Load and process discourse posts
        # with open(discourse_file, 'r', encoding='utf-8') as f:
        all_posts = []
        with open(discourse_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    print(f"Skipping empty line {idx}")
                    continue
                try:
                    all_posts.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {idx}: {e}")


        grouped = defaultdict(list)
        for post in all_posts:
            topic_id = post.get("topic_id")
            post['post_number'] = self.extract_post_number(post['url'])
            grouped[topic_id].append(post)

        for topic_posts in grouped.values():
            topic_posts.sort(key=lambda x: x['post_number'])
            self.documents.extend(self.merge_adjacent_posts(topic_posts))

    def build_index(self):
        texts = [doc.content for doc in self.documents]
        self.embeddings = self.embedder.encode(texts, show_progress_bar=True)
        dim = self.embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype('float32'))

    def search_similar(self, query: str, top_k: int = 10) -> List[Document]:
        query_vec = self.embedder.encode([query])[0].astype('float32')
        scores, indices = self.index.search(np.array([query_vec]), top_k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]

    # Original One ##########################
    # def answer_question(self, query: str) -> dict:
    #     similar_docs = self.search_similar(query)
    #     if not similar_docs:
    #         return {"answer": "No relevant information found.", "source_urls": []}

    #     context = "\n---\n".join(doc.content for doc in similar_docs)

    #     from openai import OpenAI
    #     client = OpenAI()
    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         temperature=0,
    #         messages=[
    #             {"role": "system", "content": "You are a helpful TA for a Data Science course. Answer only based on the given context. If unsure, say 'No information available'."},
    #             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    #         ]
    #     )
    #     answer = response.choices[0].message.content.strip()
    #     source_urls = [doc.url for doc in similar_docs]
    #     return {"answer": answer, "source_urls": source_urls}

    # New One
    def answer_question(self, query: str) -> dict:
        similar_docs = self.search_similar(query)
        if not similar_docs:
            return {"answer": "No relevant information found.", "source_urls": []}

        base_doc = similar_docs[0]

        # Try to find a follow-up TA reply
        ta_followup = self.get_followup_ta_post(base_doc)

        if ta_followup:
            context_docs = [ta_followup]
        else:
            context_docs = similar_docs

        context = "\n---\n".join(doc.content for doc in context_docs)

        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-3.5-turbo use gpt-4o-mini if required
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful TA for a Data Science course. Answer only based on the given context and if some numbers or keywords like score being 110 etc is mentioned in the context then state that as it is in the answer. Don't think it is a spelling mistake or something. Also try to give as much information as possible from the context. If unsure, say 'No information available'."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ]
        )

        answer = response.choices[0].message.content.strip()
        source_urls = [doc.url for doc in context_docs]

        return {"answer": answer, "source_urls": source_urls}


# FastAPI app and endpoint
app = FastAPI()

# Allow CORS for all origins (or restrict as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

pipeline = RagPipeline(
    course_file="structured_dataset.jsonl",
    discourse_file="cleaned_posts.jsonl"
)

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        result = pipeline.answer_question(request.question)

        print("Pipeline result:", result)

        links = []
        for doc_url, doc in zip(result.get("source_urls", []), pipeline.search_similar(request.question)):
            if doc.source_type == 'discourse':
                topic_id = doc.metadata.get('topic_id', 'unknown')
                url = pipeline.reconstruct_discourse_url(doc_url, topic_id)
            else:
                url = doc_url
            links.append({"url": url, "text": "Source"})

        return {
            "answer": result.get("answer", "No answer found."),
            "links": links
        }

    except Exception as e:
        print("Exception occurred:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Server error")



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "TDS Virtual TA is running"}

@app.get("/")
async def root():
    """Root endpoint with usage instructions"""
    return {
        "message": "TDS Virtual TA API", 
        "endpoints": {
            "POST /ask": "Answer questions",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "usage": "Send POST requests to /ask with JSON: {\"question\": \"your question\", \"image\": \"optional base64 image\"}"
    }

if __name__ == "__main__":
    # For local testing
    uvicorn.run(app, host="0.0.0.0", port=5000)
