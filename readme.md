# 🧠 AI Agent Prototype - (Loan Assistant)

An AI-powered **loan assistance backend** built with **FastAPI**, **LangChain**, **FAISS**, and **OpenAI embeddings**. This service ingests internal documents, builds a persistent vector store, and exposes an API that can answer user questions using Retrieval-Augmented Generation (RAG).

The project is containerized with **Docker** and designed so embeddings are computed **once** and reused across restarts.

---

## 🚀 High-Level Architecture

```
User Request
    ↓
FastAPI API (/query)
    ↓
Retriever (FAISS Vector Store)
    ↓
Relevant Documents
    ↓
LLM (OpenAI)
    ↓
Final Answer
```

---

## 📁 Project Structure

```
.
├── backend/
│   └── app/
│       ├── main.py          # FastAPI entrypoint
│       ├── retriever.py     # Document loading + FAISS logic
│       ├── chains.py        # RAG / QA chain logic
│       └── config.py        # App configuration
│
├── docs/                    # Source knowledge documents (.txt, .md)
│   ├── loan_policy.md
│   └── faq.txt
│
├── vectorstore/
│   └── faiss/               # Persisted FAISS index (generated)
│       ├── index.faiss
│       └── index.pkl
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧩 How the App Works (Step-by-Step)

### Step 0 – Startup Initialization
When the FastAPI app starts:

1. The **Retriever** is initialized
2. It checks whether a persisted FAISS index exists
3. If it exists → load it from disk
4. If not → load documents → create embeddings → build FAISS → save it

This guarantees embeddings run **only once**.

---

### Step 1 – Document Ingestion

Documents are loaded from:
```
/app/docs
```

Supported formats:
- `.txt`
- `.md`

Each document is:
- Read from disk
- Converted into LangChain `Document` objects
- Split into chunks (if configured)

---

### Step 2 – Embedding & Vector Store (One-Time)

```python
FAISS.from_documents(docs, embeddings)
```

- Uses **OpenAI Embeddings** (`text-embedding-3-small`)
- Converts text → vectors
- Stores them in FAISS

Persisted via:
```python
vectorstore.save_local("vectorstore/faiss")
```

---

### Step 3 – Query Flow (Runtime)

1. User sends a question to the API
2. Question is embedded
3. FAISS performs similarity search
4. Top-K documents are retrieved
5. Documents + question are sent to the LLM
6. Model generates a grounded response

---

## 🐳 Running with Docker

### 1️⃣ Build the Image

```bash
docker build -t ai-loan-assistant .
```

---

### 2️⃣ Run the Container (With Persistence)

```bash
docker run \
  -p 8000:8000 \
  -v $(pwd)/vectorstore:/app/vectorstore \
  --env-file .env \
  ai-loan-assistant
```

📌 The volume mount ensures embeddings **do not rerun** on restart.

---

### 3️⃣ Environment Variables (`.env`)

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

---

## 🔍 Testing the App

### Health Check

```bash
curl http://localhost:8000/health
```

### Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What loan products do we offer?"}'
```

---

## 💾 FAISS Persistence Details

FAISS data is stored at:
```
/app/vectorstore/faiss/
```

Files:
- `index.faiss` → Vector index
- `index.pkl` → Metadata + documents

If these files exist, embeddings **will not run again**.

---

## ⚠️ Common Issues

### No Documents Loaded

Log:
```
No documents found. Skipping index build.
```

Fix:
- Ensure `.txt` or `.md` files exist in `/docs`
- Confirm Docker copied or mounted the directory

---

### OpenAI Rate Limit / Quota Error

```
openai.RateLimitError: insufficient_quota
```

Fix:
- Verify billing is enabled
- Ensure embeddings are not re-running

---

## 🔐 Security Notes

- Never commit `.env` files
- API keys are injected at runtime
- Vectorstore files contain embeddings, not raw secrets

---

## 🛣️ Roadmap

- [ ] Add document hashing to auto-rebuild index only when docs change
- [ ] Add streaming responses
- [ ] Support PDF ingestion
- [ ] Add authentication
- [ ] Swap to local embeddings for zero cost

---

## 🤝 Contributing

Pull requests welcome. Please:
- Follow existing structure
- Add logging for startup steps
- Keep embeddings deterministic

---

## 📜 License

MIT License

