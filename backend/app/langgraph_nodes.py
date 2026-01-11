from typing import TypedDict, List, Optional
from langchain.schema import Document

from backend.app.llm_client import LLMClient
from backend.app.retriever import LoanKnowledgeRetriever
from backend.app.utils import get_logger
from backend.app.promts import (
  CLASSIFY_QUERY_PROMT,
  VALIDATE_RETRIEVAL_SYSTEM_RPOMT,
  VALIDATE_RETRIEVAL_USER_RPOMT,
  FORMAT_ANSWER_PROMT
)

logger = get_logger("langgraph")

# ===============================
# 1. Graph State Definition
# ===============================
class AgentState(TypedDict):
  user_query: str
  intent: Optional[str]
  retrieved_docs: List[Document]
  validated_answer: Optional[str]
  final_answer: Optional[str]
  escalate_to_human: bool


# ===============================
# 2. Initialize Shared Resources
# ===============================
llm = LLMClient()
retriever = LoanKnowledgeRetriever()


# ===============================
# 3. Query Classification Node
# ===============================
def classify_query(state: AgentState) -> AgentState:
  """
  Classifies user intent:
  - informational
  - eligibility
  - rate
  - document
  - unsupported
  """
  logger.info(
    "Classifying user query",
    extra={"extra_data": {"user_query": state["user_query"]}}
  )

  promt = CLASSIFY_QUERY_PROMT.format(
    query=state["user_query"]
  )

  intent = llm.simple_query(promt).strip().lower()

  state["intent"] = intent

  logger.info(
    "Query classified",
    extra={"extra_data": {"intent": intent}}
  )

  return state

def route_after_classification(state: AgentState) -> str:
  """
  Decide whether to retrieve knowledge based on intent.
  """
  if state["intent"] in {
    "informational",
    "eligibility",
    "rate",
    "document"
  }:
    return "retrieve"

  # unsupported or unknown intent
  return "format"


# ===============================
# 4. Knowledge Retrieval Node
# ===============================
def retrieve_knowledge(state: AgentState) -> AgentState:
  """
  Retrieves relevant policy documents using RAG.
  """

  docs = retriever.retrieve(state["user_query"], K = 4)
  state["retrieved_docs"] = docs
  return state


# ===============================
# 5. Retrieval Validation Node
# ===============================
def validate_retrieval(state: AgentState) -> AgentState:
  """
  Ensures retrieved content is sufficient.
  If not, escalate to human.
  """

  if not state["retrieved_docs"]:
    logger.warning(
      "No documents retrieved",
      extra={"extra_data": {"escalate_to_human": True}}
    )

    state["escalate_to_human"] = True
    state["validated_answer"] = None
    return state

  context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])

  user_promt = VALIDATE_RETRIEVAL_USER_RPOMT.format(
    context=context,
    query=state["user_query"]
  )

  answer = llm.simple_query(
    promt=user_promt,
    system_prompt=VALIDATE_RETRIEVAL_SYSTEM_RPOMT
  )

  if "INSUFFICIENT_CONTEXT" in answer:
    logger.warning(
      "Insufficient context for answer",
      extra={"extra_data": {"escalate_to_human": True}}
    )

    state["escalate_to_human"] = True
    state["validated_answer"] = None
  else:
    logger.info(
      "Answer validated",
      extra={"extra_data": {"escalate_to_human": False}}
    )

    state["escalate_to_human"] = False
    state["validated_answer"] = answer

  return state

def route_after_validation(state: AgentState) -> str:
  """
  Decide whether to escalate or finalize.
  """
  if state["escalate_to_human"]:
    return "format"

  # Right now both routes go to format_answer,
  # but this keeps the graph extensible (audit, retry, human queue, etc.)
  return "format"


# ===============================
# 6. Answer Formatting Node
# ===============================
def format_answer(state: AgentState) -> AgentState:
  """
  Formats safe, customer-facing response.
  """

  if state["escalate_to_human"]:
    state["final_answer"] = (
      "Thanks for your question. A loan specialist will review your request and follow up with you shortly to ensure accuracy."
    )
    return state

  promt = FORMAT_ANSWER_PROMT.format(
    answer=state["validated_answer"]
  )

  state["final_answer"] = llm.simple_query(promt)

  return state