# Streamlit application entry point
import streamlit as st
from agent.agent import get_graph, get_tools_resources
# AgentState is not directly instantiated here but its structure is used when accessing current_state fields.
# from cheese_chatbot.agent.state import AgentState 
from dotenv import load_dotenv
from langgraph.types import Command
import os
import logging
from typing import Dict, Any, Optional

# Import AIMessage if it's a specific type we need to check for, otherwise handle generically
# from langchain_core.messages import AIMessage 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Function to Extract Content ---
def extract_content_from_message(message_output) -> str:
    """Extracts string content from various message types."""
    if isinstance(message_output, str):
        return message_output
    # Check if it's an AIMessage or similar object with a 'content' attribute
    if hasattr(message_output, 'content') and isinstance(getattr(message_output, 'content'), str):
        return getattr(message_output, 'content')
    # Fallback for other types or if content is not a string
    return str(message_output)

def format_chat_history(messages: list) -> list:
    """Formats chat history for the agent's consumption."""
    if len(messages) <= 1:  # Only has initial greeting or is empty
        return []
    
    formatted_history = []
    for msg in messages[:-1]:  # Exclude current user message
        role = "user" if msg["role"] == "user" else "assistant"
        formatted_history.append((role, msg["content"]))
    return formatted_history

def update_reasoning_display(placeholder, current_state: Dict[str, Any]) -> None:
    """Updates the reasoning process display."""
    reasoning_md_parts = []
    
    if current_state.get("current_node"):
        reasoning_md_parts.append(f"**Current Node:** `{current_state['current_node']}`")
    
    if current_state.get("reasoning_steps"):
        steps_str = "\n".join([f"- {s}" for s in current_state["reasoning_steps"]])
        reasoning_md_parts.append(f"**Reasoning Log:**\n{steps_str}")
    
    if reasoning_md_parts:
        placeholder.markdown("\n\n".join(reasoning_md_parts))
    else:
        placeholder.markdown("Processing...")

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="🧀 Cheese Chatbot",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Agent Initialization ---
@st.cache_resource # Cache the agent graph, resources, and graph image
def initialize_agent_and_image(): # Renamed
    """Initializes the agent graph, resources, and generates a PNG image of the graph."""
    try:
        resources = get_tools_resources() 
        graph_instance = get_graph(resources) # get_graph from agent.py returns a CompiledGraph
        
        graph_png_bytes = None
        try:
            with open("cheese_chatbot/static/graph.png", "rb") as f:
                graph_png_bytes = f.read()
        except FileNotFoundError:
            logger.warning("Graph visualization file not found at cheese_chatbot/static/graph.png")
        except Exception as e:
            logger.error(f"Error reading graph visualization: {e}")
            graph_png_bytes = str(e)
        
        return graph_instance, graph_png_bytes
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        st.error(f"Failed to initialize the chatbot. Please check the logs and try again. Error: {e}")
        raise

# Initialize agent and graph image
try:
    app_graph, graph_image_bytes = initialize_agent_and_image()
except Exception as e:
    st.error("Critical error: Failed to initialize the chatbot. Please contact support.")
    st.stop()

# --- UI & Chat Logic ---
st.title("🧀 Kimelo's Cheese Chatbot")
st.caption("I can help you find the perfect cheese from our selection!")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! How can I help you find the perfect cheese today?"
    }]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Store current graph state for human-in-the-loop interactions
if "graph_config" not in st.session_state:
    st.session_state.graph_config = None # Will store the config to resume graph if needed

config = {"configurable": {"thread_id": "1"}}

# Main interaction loop
if prompt := st.chat_input("What cheese are you looking for? Or type 'quit' to exit."):
    if prompt.lower() == 'quit':
        st.session_state.messages.extend([
            {"role": "user", "content": "quit"},
            {"role": "assistant", "content": "Goodbye! Feel free to chat again anytime."}
        ])
        with st.chat_message("user"):
            st.markdown("quit")
        with st.chat_message("assistant"):
            st.markdown("Goodbye! Feel free to chat again anytime.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Setup reasoning process display
    with st.expander("💡 Reasoning Process", expanded=True):
        reasoning_placeholder = st.empty()
        reasoning_placeholder.markdown("Initializing response...")

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Prepare input state
            current_graph_state = None
            
            if st.session_state.graph_config and st.session_state.graph_config.get("human_feedback_needed"):
                # Handle clarification response
                current_graph_state = st.session_state.graph_config.copy()
                current_graph_state["human_response"] = prompt
                app_graph.stream(Command(resume={"human_response": prompt}), config)
            else:
                # Handle new query
                chat_history = format_chat_history(st.session_state.messages)
                current_graph_state = {
                    "input_query": prompt,
                    "chat_history": chat_history,
                    "reasoning_steps": []
                }

            # Process through graph
            for event_output in app_graph.stream(current_graph_state, config, stream_mode="values"):
                try:
                    if event_output.get('__interrupt__'):
                        current_state_output = event_output['__interrupt__'][0].value['state']
                    else:
                        current_state_output = event_output
                    
                    # Update reasoning display
                    update_reasoning_display(reasoning_placeholder, current_state_output)
                    print("="*100)
                    print("current_state_output: ", current_state_output.get("current_node"))
                    print("="*100)
                    
                    # Handle interrupts for human feedback
                    if current_state_output.get("human_feedback_needed"):
                        st.session_state.graph_config = current_state_output
                        clarification_q = current_state_output.get(
                            "clarification_question",
                            "I need more information. Could you please clarify?"
                        )
                        full_response = extract_content_from_message(clarification_q)
                        response_placeholder.markdown(full_response + " (waiting for your feedback...)")

                    # Update response if available
                    if current_state_output.get("llm_response") and current_state_output.get("current_node") == "generate_composite_response":
                        full_response = extract_content_from_message(current_state_output["llm_response"])
                        response_placeholder.markdown(full_response)

                except Exception as e:
                    logger.error(f"Error processing graph output: {e}")
                    continue

            # Handle completion
            if not current_state_output.get("human_feedback_needed"):
                st.session_state.graph_config = None
                if not full_response:
                    if current_state_output and current_state_output.get("llm_response"):
                        full_response = current_state_output.get("llm_response")
                    else:
                        full_response = "I've processed your request. Let me know if you need anything else!"
                    response_placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"An error occurred while processing your request: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            full_response = "Sorry, I encountered an error. Please try again."
            response_placeholder.markdown(full_response)
            st.session_state.graph_config = None

        # Update chat history
        if isinstance(full_response, str):
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar for LangSmith Tracing Information
# st.sidebar.info(
#     "**LangSmith Tracing:**\n\n"
#     "To enable LangSmith tracing for visualizing and debugging your agent, "
#     "ensure the following environment variables are set (e.g., in a `.env` file "
#     "at the root of your project or directly in your environment):\n"
#     "LANGCHAIN_TRACING_V2=true\n"
#     "LANGCHAIN_ENDPOINT=https://api.smith.langchain.com\n"
#     "LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY\n"
#     "LANGCHAIN_PROJECT=YOUR_PROJECT_NAME (e.g., cheese-chatbot)\n\n"
#     "Restart the Streamlit app after setting these variables."
# )
st.sidebar.markdown("---")

# Display Agent Graph Image in Sidebar
if isinstance(graph_image_bytes, bytes):
    st.sidebar.subheader("Agent Graph Structure")
    st.sidebar.image(graph_image_bytes, caption="Agent Computational Graph")
elif isinstance(graph_image_bytes, str):
    if "PLAYWRIGHT_PYTHON_MISSING" in graph_image_bytes:
        st.sidebar.warning("Graph visualization requires Playwright. Run: `pip install playwright`")
    elif "PLAYWRIGHT_SETUP_ISSUE" in graph_image_bytes:
        st.sidebar.error(f"Playwright setup issue: {graph_image_bytes.split(': ', 1)[1]}")
    else:
        st.sidebar.error(f"Graph visualization error: {graph_image_bytes}")
else: # Fallback if graph_image_bytes is None for some other reason
    st.sidebar.warning("Agent graph visualization is unavailable.")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangGraph & Streamlit.")

# TODO:
# - Add actual graph visualization if possible (e.g., using st.graphviz_chart if a DOT representation can be obtained from LangGraph)
# - Refine chat history management for the agent if specific formats (like BaseMessage objects) are required.
# - Enhance display of search results (e.g., cards for cheeses) if the agent provides structured search results in the state. 