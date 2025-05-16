from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of our cheese chatbot agent.
    """
    input_query: str
    chat_history: Annotated[list, add_messages] # Manages chat history accumulation
    query_intent: str
    
    executed_search_strategy: str# Stores the classified intent of the user's query (e.g., "count_query", "general_query")

    # Tool related fields
    search_results: List[Dict[str, Any]] # Results from hybrid_cheese_search
    
    # LLM response and reasoning
    llm_response: str
    reasoning_steps: List[str] # To store intermediate reasoning for visualization

    # Human-in-the-loop
    human_feedback_needed: bool
    clarification_question: str# Question to ask human for clarification
    human_response: str # To store the feedback provided by the human
    
    # For graph visualization and tracking
    current_node: str # Name of the node currently being executed (for visualization)
    
    status_message: str
    
    # Other potential fields can be added as needed, e.g.,
    # num_retries: int
    # error_message: str 