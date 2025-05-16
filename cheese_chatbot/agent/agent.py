import os
import uuid
import json
from typing import List, Dict, Any, TypedDict, Optional, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

from agent.state import AgentState
from data_processing.mongo_utils import get_mongo_client, get_mongo_collection, keyword_search_mongo
from data_processing.pinecone_utils import get_pinecone_index
from data_processing.embedding_utils import get_openai_embeddings_model_name # Assuming this returns the model name string
# from agent.tools.search_utils import hybrid_cheese_search # Ensure this path is correct
from langgraph.types import Command, interrupt

# Load environment variables
load_dotenv()

# Determine the absolute path to the prompts directory
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prompts')

def load_prompt_from_file(filename: str) -> str:
    """Loads a prompt string from a file in the prompts directory."""
    file_path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        # Potentially raise an error or return a default prompt if critical
        raise
    except Exception as e:
        print(f"Error reading prompt file {file_path}: {e}")
        raise

_llm_instance = None
_mongo_client_instance = None
_pinecone_client_instance = None
_embedding_model_instance = None

# --- LLM and Resource Initialization ---
def get_tools_resources(reinitialize: bool = False) -> Dict[str, Any]:
    """Initializes and returns a dictionary of shared resources."""
    global _mongo_client_instance, _pinecone_client_instance, _embedding_model_instance, _llm_instance

    if reinitialize:
        if _mongo_client_instance:
            _mongo_client_instance.close()
            _mongo_client_instance = None
        _pinecone_client_instance = None # Pinecone client v2/v3 doesn't have explicit close
        _embedding_model_instance = None
        _llm_instance = None

    if _llm_instance is None:
        _llm_instance = ChatOpenAI(model="gpt-4o", temperature=0.7)
    if _mongo_client_instance is None:
        _mongo_client_instance = get_mongo_client()
    if _pinecone_client_instance is None:
        _pinecone_client_instance = None # Explicitly set to None as it's not initialized here

    if _embedding_model_instance is None:
        embedding_model_name = get_openai_embeddings_model_name()
        _embedding_model_instance = OpenAIEmbeddings(model=embedding_model_name)

    # Directly get the Pinecone index. get_pinecone_index handles its own client initialization.
    pinecone_index_instance = None
    try:
        pinecone_index_instance = get_pinecone_index()
    except Exception as e:
        print(f"Error initializing Pinecone Index in get_tools_resources: {e}")
        # Decide if this should raise or if None is acceptable if Pinecone is optional

    return {
        "llm": _llm_instance,
        "mongo_client": _mongo_client_instance,
        "mongo_collection": get_mongo_collection(_mongo_client_instance) if _mongo_client_instance else None,
        "pinecone_index": pinecone_index_instance, # Use the directly obtained index
        "embedding_model": _embedding_model_instance,
    }

# --- Agent Nodes ---
def classify_query_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Classifies the user's query intent."""
    print(f"--- Classifying Query: {state['input_query']} ---")
    classification_llm = resources["llm"]
    # Chat history summary is no longer used by the prompt template for this node
    # chat_history_summary = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get('chat_history', [])[-5:]])

    prompt_template = load_prompt_from_file("classify_query_prompt.txt")
    # formatted_prompt = prompt_template.format(
    #     chat_history_summary=chat_history_summary, # Removed
    #     input_query=state["input_query"]
    # )
    formatted_prompt = prompt_template.format(input_query=state["input_query"])
    
    messages = [HumanMessage(content=formatted_prompt)]
    
    reasoning_steps = state.get("reasoning_steps", [])
    reasoning_steps.append(f"Classifying query: '{state['input_query']}'. Prompt: {formatted_prompt[:200]}...") # Log snippet
    
    try:
        response = classification_llm.invoke(messages)
        response_content = response.content
        
        # Clean potential markdown fences from LLM response
        if response_content.startswith("```json"):
            response_content = response_content[7:] # Remove ```json
            if response_content.endswith("```"):
                response_content = response_content[:-3] # Remove ```
        response_content = response_content.strip() # Clean up any leading/trailing whitespace
        
        # Expecting JSON response like {"intent": "greeting_query" or "information_query"}
        classification_result = json.loads(response_content)
        query_intent = classification_result.get("intent")
        
        # Validate the intent value
        if query_intent not in ["greeting_query", "unrelated_query", "cheese_query"]:
            print(f"LLM returned unexpected or missing intent: {query_intent}. Defaulting to information_query.")
            reasoning_steps.append(f"LLM returned unexpected or missing intent '{query_intent}'. Defaulting to 'information_query'.")
            query_intent = "cheese_query"
            
        print(f"Query classified as: {query_intent}")
        reasoning_steps.append(f"LLM classified intent as: {query_intent}")
        return {**state, "query_intent": query_intent, "current_node": "classify_query_node", "reasoning_steps": reasoning_steps}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from classification LLM: {e}. Content: {response.content}")
        reasoning_steps.append(f"JSONDecodeError during classification: {e}. Using default 'information_query'.")
        return {**state, "query_intent": "information_query", "current_node": "classify_query_node", "reasoning_steps": reasoning_steps} # Fallback
    except Exception as e:
        print(f"Error in classify_query_node: {e}")
        reasoning_steps.append(f"Error in classify_query_node: {e}. Using default 'information_query'.")
        return {**state, "query_intent": "information_query", "current_node": "classify_query_node", "reasoning_steps": reasoning_steps} # Fallback


def search_cheese_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Decides search strategy (MongoDB or Pinecone) using LLM and performs the search."""
    print(f"--- Determining Search Strategy for: {state['input_query']} ---")
    query = state["input_query"]
    classification_llm = resources["llm"]
    mongo_collection = resources["mongo_collection"]
    pinecone_index = resources["pinecone_index"]
    embedding_model = resources["embedding_model"]
    llm_for_mongo_keywords = resources["llm"] # Main LLM for keyword generation for Mongo

    reasoning_steps = state.get("reasoning_steps", [])
    search_results: List[Dict[str, Any]] = []
    strategy_prompt_template = load_prompt_from_file("decide_search_strategy_prompt.txt")
    formatted_strategy_prompt = strategy_prompt_template.format(input_query=query)
    
    reasoning_steps.append(f"Asking LLM to decide search strategy for query: '{query}'. Prompt: {formatted_strategy_prompt[:200]}...")

    search_strategy = "pinecone" # Default strategy
    try:
        strategy_response = classification_llm.invoke([HumanMessage(content=formatted_strategy_prompt)])
        strategy_content = strategy_response.content
        if strategy_content.startswith("```json"):
            strategy_content = strategy_content[7:-3] if strategy_content.endswith("```") else strategy_content[7:]
        strategy_content = strategy_content.strip()
        
        strategy_json = json.loads(strategy_content)
        chosen_strategy = strategy_json.get("search_strategy")
        if chosen_strategy in ["mongodb", "pinecone"]:
            search_strategy = chosen_strategy
            reasoning_steps.append(f"LLM chose search strategy: {search_strategy}")
        else:
            reasoning_steps.append(f"LLM returned invalid strategy '{chosen_strategy}'. Defaulting to {search_strategy}.")
    except Exception as e:
        print(f"Error determining search strategy with LLM: {e}. Defaulting to {search_strategy}.")
        reasoning_steps.append(f"Error determining search strategy: {e}. Defaulting to {search_strategy}.")

    print(f"--- Executing {search_strategy} search for: {query} ---")
    reasoning_steps.append(f"Executing {search_strategy} search.")

    if search_strategy == "mongodb":
        try:
            # keyword_search_mongo is expected to return a list of dicts
            # It uses an LLM (llm_for_mongo_keywords) to generate the actual mongo query.
            mongo_matches = keyword_search_mongo(mongo_collection, query, llm_for_mongo_keywords, limit=10)
            reasoning_steps.append(f"MongoDB keyword search returned {len(mongo_matches)} matches.")
            for item in mongo_matches:
                if '_id' in item:
                    item['_id'] = str(item['_id']) # Convert ObjectId
                search_results.append(item)
        except Exception as e:
            print(f"Error during MongoDB search: {e}")
            reasoning_steps.append(f"MongoDB search failed: {e}")

    elif search_strategy == "pinecone":
        try:
            query_vector = embedding_model.embed_query(query)
            reasoning_steps.append(f"Generated query vector for Pinecone semantic search.")
            
            pinecone_response = pinecone_index.query(
                vector=query_vector, 
                top_k=20, 
                include_metadata=True
            )
            reasoning_steps.append(f"Pinecone query returned {len(pinecone_response.get('matches', []))} matches.")
            
            for match in pinecone_response.get('matches', []):
                if match.get('metadata'):
                    cheese_data = dict(match.get('metadata'))
                    cheese_data['sku'] = match.get('id', cheese_data.get('sku')) # SKU is the Pinecone ID
                    cheese_data['search_score_semantic'] = match.get('score', 0.0)
                    if '_id' in cheese_data: # Should not be from Pinecone metadata ideally
                        del cheese_data['_id']
                    search_results.append(cheese_data)
                else:
                    reasoning_steps.append(f"Warning: Pinecone match ID {match.get('id')} missing metadata.")
        except Exception as e:
            print(f"Error during Pinecone semantic search: {e}")
            reasoning_steps.append(f"Pinecone search failed: {e}")
    
    # If both fail or somehow strategy is not set, search_results will be empty.
    # The hybrid_cheese_search function is still available if we want a more robust fallback.
    # For now, we stick to the chosen single strategy.

    print(f"Found {len(search_results)} results from {search_strategy} search.")
    print("Search results: ", search_results)
    reasoning_steps.append(f"{search_strategy.capitalize()} search returned {len(search_results)} items.")
    return {**state, "search_results": search_results, "executed_search_strategy": search_strategy, "current_node": "search_cheese_node", "reasoning_steps": reasoning_steps}


def evaluate_search_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Evaluates search results and decides if human clarification is needed for an information_query."""
    print("--- Evaluating Search Results ---")
    search_results = state.get("search_results", [])
    classification_llm = resources["llm"]
    input_query = state["input_query"]
    executed_search_strategy = state.get("executed_search_strategy", "unknown")
    
    print("Debuging agent.py line 231: Executed search strategy: ", executed_search_strategy)

    reasoning_steps = state.get("reasoning_steps", [])
    reasoning_steps.append(f"Evaluating {len(search_results)} search results for query: '{input_query}'. Strategy used: {executed_search_strategy}.")

    # Prepare the top N results as a JSON string for the prompt
    # Limiting to top 5 for brevity in the prompt, but can be adjusted.
    # top_results_for_prompt = search_results[:5]
    search_results_details_json = json.dumps(search_results, indent=2) # Convert to JSON string
    
    # The user previously added a print statement here for results_summary, 
    # we can update it or remove it. For now, let's print the new JSON details.
    # print(f"Debuging agent.py line 230: Search results details (JSON for prompt): {search_results_details_json}")

    prompt_template = load_prompt_from_file("evaluate_search_clarification_prompt.txt")

    formatted_prompt = prompt_template.format(
        input_query=input_query,
        search_results_details_json=search_results_details_json, # Use new placeholder and pass JSON string
        executed_search_strategy=executed_search_strategy
    )
    
    messages = [HumanMessage(content=formatted_prompt)]
    # Increased length for prompt logging due to JSON details
    reasoning_steps.append(f"Asking LLM if clarification is needed. Strategy: {executed_search_strategy}. Prompt (first 500 chars): {formatted_prompt[:500]}...")

    try:
        response = classification_llm.invoke(messages)
        response_content = response.content
        # Clean potential markdown fences
        if response_content.startswith("```json"):
            response_content = response_content[7:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
        response_content = response_content.strip()
        print("Debuging agent.py line 266: Response content: ", response_content)

        evaluation = json.loads(response_content) # Expects {"clarification_needed": bool, "clarification_question": str}
        
        human_feedback_needed = evaluation.get("clarification_needed", False)
        clarification_prompt = evaluation.get("clarification_question", "")
        
        print("Debuging agent.py line 273: Human feedback needed: ", evaluation)

        if human_feedback_needed:
            reasoning_steps.append(f"LLM suggests clarification is needed. Question: '{clarification_prompt}'")
        else:
            reasoning_steps.append("LLM suggests query is clear enough or clarification not beneficial. Proceeding to generate response.")
        
        print("Debuging agent.py line 278: Human feedback needed: ", human_feedback_needed)
        print("Debuging agent.py line 279: Clarification question: ", clarification_prompt)
        
        return {
            **state, 
            "human_feedback_needed": human_feedback_needed, 
            "clarification_question": clarification_prompt if human_feedback_needed else "",
            "current_node": "evaluate_search_node",
            "reasoning_steps": reasoning_steps
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from evaluation LLM: {e}. Content: {response.content}")
        reasoning_steps.append(f"JSONDecodeError during evaluation: {e}. Proceeding without clarification.")
        return {**state, "human_feedback_needed": False, "current_node": "evaluate_search_node", "reasoning_steps": reasoning_steps} # Fallback
    except Exception as e:
        print(f"Error in evaluate_search_node LLM call: {e}")
        reasoning_steps.append(f"Error in evaluate_search_node LLM call: {e}. Proceeding without clarification.")
        return {**state, "human_feedback_needed": False, "current_node": "evaluate_search_node", "reasoning_steps": reasoning_steps} # Fallback

def request_human_input_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Node that signifies the agent needs to wait for human input.
    It retrieves the clarification question from the state and sets a status message.
    """
    print("--- Requesting Human Input ---")
    clarification_question_from_state = state.get("clarification_question")
    reasoning_steps = state.get("reasoning_steps", [])
    
    if not clarification_question_from_state: # Handles None or empty string
        clarification_to_ask = "I'm not sure how to proceed. Could you please provide more details or rephrase your query?"
        warning_message = "Warning: Clarification question from state was empty or missing. Using default question."
        print(warning_message)
        reasoning_steps.append(warning_message)
    else:
        clarification_to_ask = clarification_question_from_state

    print(f"AGENT ASKS: {clarification_to_ask}")
    
    reasoning_steps.append(f"Agent is requesting human input. Question posed to user: \"{clarification_to_ask}\"")
    
    new_state = {
        **state, 
        "current_node": "request_human_input", 
        "status_message": f"Waiting for your response to: {clarification_to_ask}",
        "reasoning_steps": reasoning_steps
    }
    
    human_response = interrupt({"state": new_state})
    human_response = human_response.get("human_response", "")
    print("Debuging agent.py line 331: Human response: ", human_response)
    # This node itself doesn't return human_input; it signals the graph to wait.
    # The human input would then be injected into the state by an external mechanism
    # before the graph transitions to the next node (e.g., process_human_input_node).
    return {
        **state, 
        "human_response": human_response,
        "current_node": "request_human_input", 
        # "status_message": f"Waiting for your response to: {clarification_to_ask}",
        "reasoning_steps": reasoning_steps
    }


def process_human_input_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Processes the human's response to a clarification request."""
    human_response = state.get("human_response", "")
    input_query = state.get("input_query", "")
    print(f"--- Processing Human Input: {human_response} ---")
    
    chat_history = state.get("chat_history", [])
    original_query_for_clarification = state.get("original_query_for_clarification", state.get("input_query")) # Keep track of original query

    # Add the clarification interaction to history
    # The question asked should be in "clarification_question"
    assistant_question_asked = state.get("clarification_question")
    if assistant_question_asked and assistant_question_asked.strip():
        # Avoid adding if it's empty or None.
        # Check if the last assistant message was already this question to avoid duplicates if re-entering.
        if not chat_history or not (chat_history[-1].content == assistant_question_asked and chat_history[-1].type == "assistant_clarification"):
             chat_history.append(("ai", assistant_question_asked))
    
    # Ensure human_response is not empty before adding
    if human_response and human_response.strip():
        chat_history.append(("user", human_response))

    reasoning_steps = state.get("reasoning_steps", [])
    reasoning_steps.append(f"Human provided response: '{human_response}'. This will be treated as the new query, and original query was: '{original_query_for_clarification}'.")

    # The human's response becomes the new input_query
    # Store the original query that led to this, in case it's needed for context later (though chat history also has it)
    return {
        **state,
        "input_query": input_query + " " + human_response, 
        "original_query_for_clarification": original_query_for_clarification, # Persist original query if needed
        "chat_history": chat_history,
        "search_results": [],
        "executed_search_strategy": None, # Reset strategy as it's a new query context
        "llm_response": None,
        "human_feedback_needed": False, # Reset flag
        "clarification_question": "",   # Clear the question
        "human_response": "",           # Clear the processed response
        "current_node": "process_human_input_node",
        "needs_new_search": True,       # Flag that a new search cycle should start
        "reasoning_steps": reasoning_steps
    }


def generate_composite_response_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Generates a comprehensive response to the user based on search results and query type using a composite prompt."""
    print("--- Generating Composite Response ---")
    llm = resources["llm"]
    input_query = state["input_query"]
    search_results = state.get("search_results", [])
    chat_history = state.get("chat_history", [])
    executed_search_strategy = state.get("executed_search_strategy", "unknown")

    reasoning_steps = state.get("reasoning_steps", [])
    reasoning_steps.append(f"Generating composite response for query: '{input_query}'. Strategy used: {executed_search_strategy}")

    # chat_history_for_prompt_summary is no longer used by this node's prompt.
    # chat_history_for_prompt_summary = "\n".join([
    #     f"{msg['role'] if isinstance(msg, dict) else (msg[0] if isinstance(msg, tuple) else msg.type)}: {msg['content'] if isinstance(msg, dict) else (msg[1] if isinstance(msg, tuple) else msg.content)}"
    #     for msg in chat_history[-5:] # Last 5 messages for summary
    # ])
    # if not chat_history_for_prompt_summary:
    #     chat_history_for_prompt_summary = "No recent chat history."

    # Step 1: Prepare search_results_summary_for_llm by serializing all search results to JSON
    search_results_summary_for_llm = "No relevant cheeses found in the search."

    if search_results:
        try:
            search_results_summary_for_llm = json.dumps(search_results) 
            reasoning_steps.append(f"Serialized all {len(search_results)} search results to JSON for the main prompt.")
        except TypeError as e:
            reasoning_steps.append(f"TypeError during JSON serialization of search results: {e}. Falling back to basic list.")
            # Fallback if complex objects are not serializable by default json.dumps, though this should be rare with dicts
            search_results_summary_for_llm = str(search_results) 
    else:
        reasoning_steps.append("No search results to summarize for the main prompt.")

    count = len(search_results)
    # Step 2: Call the main LLM to generate the composite response
    system_prompt_template_name = "generate_composite_response_prompt.txt"
    prompt_template = load_prompt_from_file(system_prompt_template_name)
    
    formatted_system_prompt = prompt_template.format(
        input_query=input_query,
        # chat_history_summary=chat_history_for_prompt_summary, # Removed
        search_results_summary=search_results_summary_for_llm,
        executed_search_strategy=executed_search_strategy,
        count=count if count > 0 else "undefined", # Ensure count is properly handled if 0, prompt expects a string or number not literally "undefined"
        # result_type_classification is removed
    )
    
    reasoning_steps.append(f"Using composite response prompt. Summary/Data: {search_results_summary_for_llm[:200]}...")

    user_message_content = input_query 

    messages = [
        HumanMessage(content=formatted_system_prompt) 
    ]
    reasoning_steps.append(f"System Prompt Snippet ({system_prompt_template_name}): {formatted_system_prompt[:300]}...")
    reasoning_steps.append(f"User Message to LLM: {user_message_content[:200]}...")

    try:
        response = llm.invoke(messages)
        final_response = response.content
        print(f"LLM Composite Response: {final_response}")
        reasoning_steps.append(f"LLM generated composite response: {final_response[:100]}...")
        
        updated_chat_history = list(chat_history)
        if not updated_chat_history or updated_chat_history[-1].content != input_query or updated_chat_history[-1].type != "user":
            if input_query and input_query.strip():
                 updated_chat_history.append(("user", input_query))
        updated_chat_history.append(("assistant", final_response))

        return {**state, "llm_response": final_response, "chat_history": updated_chat_history, "current_node": "generate_composite_response_node", "reasoning_steps": reasoning_steps}
    except Exception as e:
        print(f"Error in generate_composite_response_node: {e}")
        reasoning_steps.append(f"Error during composite response generation: {e}")
        fallback_response = "I encountered an issue while generating a response. Please try again."
        return {**state, "llm_response": fallback_response, "chat_history": chat_history, "current_node": "generate_composite_response_node", "reasoning_steps": reasoning_steps}

# --- New Node for Greeting ---
def generate_greeting_response_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Generates a simple greeting response."""
    print("--- Generating Greeting Response ---")
    reasoning_steps = state.get("reasoning_steps", [])
    reasoning_steps.append("Generating a static greeting response.")
    
    chat_history = state.get("chat_history", [])
    if not chat_history or chat_history[-1].content != state.get("input_query") or chat_history[-1].type != "user":
        chat_history.append(("user", state.get("input_query","")))

    greeting_response = "Hello there! I'm Kimelo's Cheese Chatbot. How can I help you find the perfect cheese today?"
    chat_history.append(("assistant", greeting_response))

    return {
        **state, 
        "llm_response": greeting_response, 
        "chat_history": chat_history,
        "current_node": "generate_greeting_response_node", 
        "reasoning_steps": reasoning_steps
    }

# --- New Node for Unrelated Queries ---
def handle_unrelated_query_node(state: AgentState, resources: Dict[str, Any]) -> AgentState:
    """Generates a response for unrelated queries."""
    print("--- Handling Unrelated Query ---")
    reasoning_steps = state.get("reasoning_steps", [])
    reasoning_steps.append("Generating a static response for an unrelated query.")
    
    chat_history = state.get("chat_history", [])
    # Avoid duplicating if this node is somehow re-entered without history update
    if not chat_history or chat_history[-1].content != state.get("input_query") or chat_history[-1].type != "user":
        if state.get("input_query","").strip(): # Ensure input_query is not empty/whitespace before adding
            chat_history.append(("user", state.get("input_query","")))

    unrelated_response = "I am Kimelo's Cheese Chatbot. I can help you with questions about cheese, but I can't help with that. Is there anything cheese-related I can assist you with?"
    chat_history.append(("assistant", unrelated_response))

    return {
        **state,
        "llm_response": unrelated_response,
        "chat_history": chat_history,
        "current_node": "handle_unrelated_query_node",
        "reasoning_steps": reasoning_steps
    }

# --- Graph Definition ---
def get_graph(resources: Dict[str, Any]):
    """Builds and compiles the LangGraph for the cheese chatbot."""
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("classify_query", lambda state: classify_query_node(state, resources))
    graph_builder.add_node("search_cheese", lambda state: search_cheese_node(state, resources))
    graph_builder.add_node("evaluate_search", lambda state: evaluate_search_node(state, resources))
    graph_builder.add_node("request_human_input", lambda state: request_human_input_node(state, resources))
    graph_builder.add_node("process_human_input", lambda state: process_human_input_node(state, resources))
    graph_builder.add_node("generate_composite_response", lambda state: generate_composite_response_node(state, resources))
    graph_builder.add_node("generate_greeting_response", lambda state: generate_greeting_response_node(state, resources))
    graph_builder.add_node("handle_unrelated_query", lambda state: handle_unrelated_query_node(state, resources)) # New node

    # Set entry point
    graph_builder.add_edge(START, "classify_query")

    # Define edges
    def route_after_classification(state: AgentState):
        query_intent = state.get("query_intent")
        if query_intent == "greeting_query":
            return "generate_greeting_response"
        elif query_intent == "cheese_query": # Changed from information_query
            return "search_cheese"
        elif query_intent == "unrelated_query":
            return "handle_unrelated_query" # Route to new handler
        else:
            # Fallback, though classify_query_node should provide a default
            print(f"Unknown intent in routing: {query_intent}. Defaulting to search_cheese.") 
            return "search_cheese"

    graph_builder.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "generate_greeting_response": "generate_greeting_response",
            "search_cheese": "search_cheese",
            "handle_unrelated_query": "handle_unrelated_query" # Added new route
        }
    )
    graph_builder.add_edge("search_cheese", "evaluate_search")

    # Conditional edge from evaluate_search
    graph_builder.add_conditional_edges(
        "evaluate_search",
        lambda state: "request_human_input" if state.get("human_feedback_needed") else "generate_composite_response",
        {
            "request_human_input": "request_human_input",
            "generate_composite_response": "generate_composite_response"
        }
    )

    graph_builder.add_edge("request_human_input", "process_human_input")
    # After processing human input, always re-classify the (potentially refined) query
    graph_builder.add_edge("process_human_input", "classify_query") 
    
    # Define terminal nodes
    graph_builder.add_edge("generate_composite_response", END)
    graph_builder.add_edge("generate_greeting_response", END)
    graph_builder.add_edge("handle_unrelated_query", END) # New edge to END

    memory = MemorySaver()

    compiled_graph = graph_builder.compile(checkpointer=memory)
    return compiled_graph