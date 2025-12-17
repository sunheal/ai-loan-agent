from typing import TypedDict, List, Optional

from langchain.schema import Document
from backend.app.llm_client import LLMClient
from backend.app.retriever import LoanKnowledgeRetriever


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

  promt = f"""
  Classify the following loan-related user query into one category:

  Categories:
  - informational
  - eligibility
  - rate
  - document
  - unsupported

  Query: "{state['user_query']}"

  Return only the category name
  """

  intent = llm.complete(promt).strip().lower()

  state["intent"] = intent
  return state


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
    state["escalate_to_human"] = True
    state["validated_answer"] = None
    return state

  context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])

  promt = f"""
  Using ONLY the context below, answer the user's question.
  If the answer is not fully supported by the context, respond with "INSUFFICIENT_CONTEXT".

  Context: {context}

  Question: {state["user_query"]}
  """

  answer = llm.complete(promt)

  if "INSUFFICIENT_CONTEXT" in answer:
    state["escalate_to_human"] = True
    state["validated_answer"] = None
  else:
    state["escalate_to_human"] = False
    state["validated_answer"] = answer

  return state


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

  promt = f"""
  Rewrite the following answer in a clear, professional, customer-friendly tone. Do NOT add new information.PermissionError

  Answer: {state['validated_answer']}
  """

  formatted = llm.complete(promt)
  state["final_answer"] = formatted

  return state