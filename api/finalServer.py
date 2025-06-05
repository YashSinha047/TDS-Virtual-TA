import json
import re
import traceback
from collections import defaultdict
from typing import List, Optional
from dotenv import load_dotenv
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import uvicorn
import pickle
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Document structure for storage and indexing
class Document:
    def __init__(self, content, url, source_type, metadata=None):
        self.content = content
        self.url = url
        self.source_type = source_type
        self.metadata = metadata or {}

# FastAPI request/response models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

# Core RAG pipeline
class RagPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index("faiss.index")
        with open("documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def extract_post_number(self, url):
        match = re.search(r'/([0-9]+)$', url)
        return int(match.group(1)) if match else 0

    def get_best_followup_ta_post(self, query: str, base_doc: Document) -> Optional[Document]:
        topic_id = base_doc.metadata.get("topic_id")
        base_post_number = self.extract_post_number(base_doc.url)
        ta_usernames = {'Jivraj', 'carlton', 's.anand'}

        candidates = [
            doc for doc in self.documents
            if doc.source_type == 'discourse'
            and doc.metadata.get("topic_id") == topic_id
            and self.extract_post_number(doc.url) > base_post_number
            and doc.metadata.get("username") in ta_usernames
        ]

        if not candidates:
            return None

        query_embedding = self.embedder.encode([query])[0]
        candidate_embeddings = self.embedder.encode([doc.content for doc in candidates])
        similarities = [np.dot(query_embedding, emb) for emb in candidate_embeddings]

        best_index = int(np.argmax(similarities))
        if similarities[best_index] < 0.5:
            return None
        return candidates[best_index]

    def reconstruct_discourse_url(self, original_url: str, topic_id: str) -> str:
        base_url = "https://discourse.onlinedegree.iitm.ac.in"
        m = re.search(r'/t/([^/]+)/([0-9]+)$', original_url)
        if not m:
            return original_url
        slug_with_id = m.group(1)
        parts = slug_with_id.rsplit('-', 1)
        slug = parts[0] if len(parts) == 2 and parts[1].isdigit() else slug_with_id
        if any(month in slug.lower() for month in ['jan', 'feb', 'mar', 'apr']) and '2025' not in slug:
            slug += '-2025'
        return f"{base_url}/t/{slug}/{topic_id}"

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
            merged_docs.append(Document(content=merged_content, url=metadata['url'], source_type='discourse', metadata=metadata))
        return merged_docs

    def load_data(self, course_file: str, discourse_file: str):
        with open(course_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.documents.append(Document(content=data['content'], url=data['url'], source_type='course'))

        all_posts = []
        with open(discourse_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_posts.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {idx}: {e}")

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

    def search_similar(self, query: str, top_k: int = 3) -> List[Document]:
        query_vec = self.embedder.encode([query])[0].astype('float32')
        scores, indices = self.index.search(np.array([query_vec]), top_k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]

    def answer_question(self, query: str) -> dict:
        similar_docs = self.search_similar(query)
        if not similar_docs:
            return {"answer": "No relevant information found.", "source_urls": []}

        base_doc = similar_docs[0]
        ta_followup = self.get_best_followup_ta_post(query, base_doc)
        context_docs = [ta_followup] if ta_followup else similar_docs
        context = "\n---\n".join(doc.content for doc in context_docs)

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful TA for a Data Science course. Answer only based on the given context. If unsure, say 'No information available'."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ]
        )
        answer = response.choices[0].message.content.strip()
        source_urls = [doc.url for doc in context_docs]
        return {"answer": answer, "source_urls": source_urls}

# FastAPI app
app = FastAPI()
pipeline = RagPipeline()

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        result = pipeline.answer_question(request.question)
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
    return {"status": "healthy", "message": "TDS Virtual TA is running"}

@app.get("/")
async def root():
    return {
        "message": "TDS Virtual TA API",
        "endpoints": {
            "POST /ask": "Answer questions",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "usage": "Send POST requests to /ask with JSON: {\"question\": \"your question\"}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
