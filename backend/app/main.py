from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.models import ChatRequest, ChatResponse
from backend.app.orchestrator import LoanAssistantOrchestrator
from backend.app.retriever import LoanKnowledgeRetriever

import uuid
from backend.app.utils import get_logger

logger = get_logger("api")

# ===============================
# 1. FastAPI App Initialization
# ===============================
app = FastAPI(
  title="AI Loan Assistant",
  description="LLM-powered loan assistant with RAG, LangChain and LangGraph",
  version="1.0.0"
)

# ===============================
# 2. CORS Configuration
# ===============================
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"], # restrict in production
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

# ===============================
# 3. Initialize Core Components
# ===============================
orchestrator = LoanAssistantOrchestrator()
retriever = LoanKnowledgeRetriever()

# ===============================
# 4. Startup Event
# ===============================
@app.on_event("startup")
def startup_event():
  """
  Build the vector index once when the app starts.
  """
  retriever.build_index()

# ===============================
# 5. Health Check Endpoint
# ===============================
@app.get("/health")
def health_check():
  return {"status": "ok"}

# ===============================
# 6. Chat Endpoint
# ===============================
@app.post("/chat", response_model=ChatResponse)
def chat(reuqest: ChatRequest):
  """
  Main entry point for the AI Loan Assistant.
  """
  request_id = str(uuid.uuid4())

  logger.info(
    "Incoming chat request",
    extra={
      "extra_data": {
        "request_id": request_id,
        "message_length": len(reuqest.message)
      }
    }
  )

  answer = orchestrator.run(reuqest.message)

  logger.info(
    "Chat request completed",
    extra={
      "extra_data": {
        "request_id": request_id
      }
    }
  )

  return ChatResponse(answer=answer)