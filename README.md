TDS Virtual TA -- IITM BS May 2025
=================================

A **Retrieval-Augmented Generation (RAG)-based Q&A API** designed as a **Virtual Teaching Assistant** for the *Tools for Data Science* (TDS) course in the **IITM BS May 2025** Term. This API enables students to ask natural language questions and receive accurate answers grounded in official course content and instructor/TA responses from the Discourse forum for Jan 2025 Term.

* * * * *

🚀 Features
-----------

-   🔍 **Embedding-based Semantic Search**: Queries course material and Discourse posts for relevant content.
-   🎯 **Prioritized TA Responses**: Preference given to TA answers following matched student posts.
-   🧵 **Topic-wise Contextual Analysis**: Scans discussions for contextual accuracy.
-   📤 **`/ask` API Endpoint**: Handles user queries with natural language processing.
-   🌐 **CORS-enabled Public API**: Accessible for cross-origin requests.
-   📄 **Source Attribution**: Returns source links with every answer for transparency.

* * * * *

🛠️ Tech Stack
--------------

-   **FastAPI**: Lightweight and fast web server for the API.
-   **sentence-transformers**: Generates embeddings for semantic search.
-   **FAISS**: Efficient vector similarity search for retrieval.
-   **OpenAI GPT**: Powers answer generation with contextual grounding.
-   **Uvicorn**: ASGI server for running the FastAPI application.
-   **dotenv**: Manages environment variables securely.

* * * * *

📦 Installation
---------------

Follow these steps to set up the project locally:

1.  **Clone the Repository**

    ```
    git clone https://github.com/YOUR_USERNAME/tds-virtual-ta.git
    cd tds-virtual-ta

    ```

2.  **Install Dependencies**

    ```
    pip install -r requirements.txt

    ```

3.  **Set Up Environment Variables**\
    Create a `.env` file in the project root and add your OpenAI API key:

    ```
    OPENAI_API_KEY=your_openai_api_key_here

    ```

* * * * *

▶️ Running Locally
------------------

1.  **Start the FastAPI Server**

    ```
    python -m uvicorn tds_virtual_ta_api:app --host 0.0.0.0 --port 5000

    ```

2.  **Enable Public Access (Optional)**\
    For public access (e.g., for submission), use ngrok:

    ```
    ./ngrok.exe http 5000

    ```

    This will provide a public URL, e.g.:

    ```
    https://your-subdomain.ngrok-free.app

    ```

* * * * *

🔁 API Endpoints
----------------

### **POST /ask**

Ask a question and receive a context-grounded answer.

**Request Format**:

```
{
  "question": "What is the time duration of ROE exam?",
  "image": "optional base64 image"
}

```

**Response Format**:

```
{
  "answer": "ROE exam is generally of 45 minutes",
  "links": [
    {"url": "https://yourcourse.org/topic/123/5", "text": "Source"}
  ]
}

```

### **GET /health**

Basic health check endpoint.

**Response**:

```
{
  "status": "healthy",
  "message": "TDS Virtual TA is running"
}

```

### **GET /**

Root endpoint with usage and metadata.

**Response**:

```
{
  "message": "TDS Virtual TA API",
  "endpoints": {
    "POST /ask": "Answer questions",
    "GET /health": "Health check",
    "GET /docs": "API documentation"
  },
  "usage": "Send POST requests to /ask with JSON: {\"question\": \"your question\", \"image\": \"optional base64 image\"}"
}

```

* * * * *

🧪 Prompt Testing
-----------------

We used **Promptfoo** to conduct systematic prompt evaluations across various question types and sources, ensuring robust performance and answer quality.

* * * * *

📁 Project Structure
--------------------

```
├── tds_virtual_ta_api.py        # FastAPI application
├── structured_dataset.jsonl     # Official course content
├── cleaned_posts.jsonl          # Processed Discourse forum posts
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (API keys)
├── LICENSE                     # MIT License
└── README.md                   # Project documentation

```

* * * * *

📄 License
----------

This project is licensed under the [MIT License].

* * * * *

📝 Notes
--------

-   Replace `YOUR_USERNAME` in the clone command with your actual GitHub username.
-   Ensure the `.env` file is not committed to version control (add it to `.gitignore`).
-   For production deployment, consider additional security measures (e.g., rate limiting, authentication).
