from langgraph.graph import StateGraph, END

from backend.app.langgraph_nodes import (
  AgentState,
  classify_query,
  retrieve_knowledge,
  validate_retrieval,
  format_answer,
  route_after_classification,
  route_after_validation
)

class LoanAssistantOrchestrator:
  """
  Orchestrates the AI Loan Assistant workflow using LangGraph.
  """

  def __init__(self):
    self.graph =self._build_graph()

  # -----------------------------
  # Build LangGraph Workflow
  # -----------------------------

  def _build_graph(self):
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("validate_retrieval", validate_retrieval)
    graph.add_node("format_answer", format_answer)

    # Define execution order
    graph.set_entry_point("classify_query")

    # Conditional branching AFTER classification
    graph.add_conditional_edges(
      "classify_query",
      route_after_classification,
      {
        "retrieve": "retrieve_knowledge",
        "format": "format_answer"
      }
    )

    # Normal edge: retrieval → validation
    graph.add_edge("retrieve_knowledge", "validate_retrieval")

    # Conditional branching AFTER validation
    graph.add_conditional_edges(
      "validate_retrieval",
      route_after_validation,
      {
        "format": "format_answer"
      }
    )

    # End graph
    graph.add_edge("format_answer", END)

  # -----------------------------
  # Public Execution Method
  # -----------------------------
  def run(self, user_query: str) -> str:
    """
    Executes the graph for a single user query.
    """

    initial_state: AgentState = {
      "user_query": user_query,
      "intent": None,
      "retrieved_docs": [],
      "validated_answer": None,
      "final_answer": None,
      "escalate_to_human": False
    }

    result = self.graph.invoke(initial_state)
    return result["final_answer"]