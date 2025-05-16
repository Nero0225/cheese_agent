# Streamlit application entry point
import streamlit as st
from agent.agent import get_graph, get_tools_resources
# AgentState is not directly instantiated here but its structure is used when accessing current_state fields.
# from cheese_chatbot.agent.state import AgentState 
from dotenv import load_dotenv
from langgraph.types import Command
import os

# Import AIMessage if it's a specific type we need to check for, otherwise handle generically
# from langchain_core.messages import AIMessage 

# --- Helper Function to Extract Content ---
def extract_content_from_message(message_output):
    """Extracts string content from various message types."""
    if isinstance(message_output, str):
        return message_output
    # Check if it's an AIMessage or similar object with a 'content' attribute
    if hasattr(message_output, 'content') and isinstance(getattr(message_output, 'content'), str):
        return getattr(message_output, 'content')
    # Fallback for other types or if content is not a string
    return str(message_output)

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="🧀 Cheese Chatbot", page_icon="🧀", layout="wide")

# --- Agent Initialization ---
@st.cache_resource # Cache the agent graph, resources, and graph image
def initialize_agent_and_image(): # Renamed
    """Initializes the agent graph, resources, and generates a PNG image of the graph."""
    resources = get_tools_resources() 
    graph_instance = get_graph(resources) # get_graph from agent.py returns a CompiledGraph
    
    graph_png_bytes = None
    try:
        with open("cheese_chatbot/static/graph.png", "rb") as f:
            graph_png_bytes = f.read()
        # CompiledGraph.get_graph() returns the StateGraph.
        # StateGraph.draw_mermaid_png() generates the image bytes.
        # state_graph_for_drawing = graph_instance.get_graph()
        # graph_png_bytes = state_graph_for_drawing.draw_mermaid()
        # with open("graph.txt", "w") as f:
        #     f.write(graph_png_bytes)
    except ImportError: # Specifically for Playwright not being installed at Python level
        # This warning will appear in the sidebar early if Playwright Python package is missing.
        # It's less ideal than a direct st.sidebar.warning, but st context might not be fully available here.
        print("Playwright Python package is not installed. Cannot generate graph image. Run `pip install playwright`")
        # We can't use st.sidebar.warning directly here as this function is cached and might run before sidebar is drawn.
        # Instead, we'll handle the None return value later.
        graph_png_bytes = "PLAYWRIGHT_PYTHON_MISSING" # Special string to indicate this specific error
    except Exception as e:
        # Catch other potential errors during PNG generation (e.g., Playwright browser binaries missing)
        # 'No node found for selector' can happen if Playwright's browser isn't installed via 'playwright install'
        error_message = f"Could not generate graph image due to: {str(e)}. Ensure Playwright is fully installed and configured ('pip install playwright' and then 'playwright install')."
        print(error_message) # Log to console
        # Similar to above, direct st.sidebar.error is tricky in cached function.
        graph_png_bytes = error_message # Store error message to display later
        if "playwright" in str(e).lower() or "node found for selector" in str(e).lower() :
             graph_png_bytes = "PLAYWRIGHT_SETUP_ISSUE: " + str(e)


    return graph_instance, graph_png_bytes

app_graph, graph_image_bytes = initialize_agent_and_image() # Updated call

# --- UI & Chat Logic ---
st.title("🧀 Kimelo's Cheese Chatbot")
st.caption("I can help you find the perfect cheese from our selection!")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you find the perfect cheese today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Store current graph state for human-in-the-loop interactions
if "graph_config" not in st.session_state:
    st.session_state.graph_config = None # Will store the config to resume graph if needed

# Main interaction loop
if prompt := st.chat_input("What cheese are you looking for? Or type 'quit' to exit."):
    if prompt.lower() == 'quit':
        st.session_state.messages.append({"role": "user", "content": "quit"})
        with st.chat_message("user"):
            st.markdown("quit")
        st.session_state.messages.append({"role": "assistant", "content": "Goodbye! Feel free to chat again anytime."})
        with st.chat_message("assistant"):
            st.markdown("Goodbye! Feel free to chat again anytime.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Moved Expander: Display reasoning steps BEFORE the assistant's chat message bubble
    with st.expander("💡 Reasoning Process", expanded=True):
        reasoning_placeholder = st.empty()
        reasoning_placeholder.markdown("Waiting for agent to start...")

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        inputs = {}
        current_graph_state_for_input = None # To hold the state that will be passed to the graph

        # Check if we are responding to a clarification request
        if st.session_state.graph_config and st.session_state.graph_config.get("human_feedback_needed"):
            # This is a response to a clarification.
            # The 'inputs' should be the *entire previous state dictionary* plus the human_response.
            # The agent's process_human_input_node will update input_query from human_response.
            current_graph_state_for_input = st.session_state.graph_config.copy()
            current_graph_state_for_input["human_response"] = prompt 
            Command(resume={"human_response": prompt})
            
            # Clear the flag now that we're processing the feedback.
            # process_human_input_node should ensure human_feedback_needed is reset to False.
            # st.session_state.graph_config["human_feedback_needed"] = False # No, let process_human_input_node handle this
            # st.session_state.graph_config = None # Consume/clear the interrupt state for this turn.
        else:
            # This is a fresh query or continuation without prior interruption.
            # Construct chat history from Streamlit's session state for the agent.
            # Agent expects chat_history as list of tuples: [("user", "Hi"), ("assistant", "Hello")]
            # or a list of BaseMessage objects if the agent is adapted for that.
            # For now, using tuples as per AgentState.
            
            # Correctly build chat_history_for_agent from st.session_state.messages
            # It should include all messages *before* the current user prompt.
            chat_history_for_agent = []
            # st.session_state.messages includes the current user prompt if just added.
            # So, history is everything *except* the last message if it's the current prompt.
            # However, st.session_state.messages *already* has the user's new prompt.
            # The agent expects history *before* the current input_query.
            
            # The agent will add the current "user" prompt and "assistant" response to its history.
            # So, pass the history *before* the current user prompt.
            # st.session_state.messages is: [..., prev_assistant_msg, current_user_prompt]
            # We need to pass [..., prev_assistant_msg]
            
            temp_history_for_agent = []
            if len(st.session_state.messages) > 1: # Has at least one message before current user input
                 for msg_obj in st.session_state.messages[:-1]: # All but the last (current user prompt)
                    role = msg_obj["role"]
                    # Ensure roles are "user" or "assistant" as expected by the agent's history format
                    # AgentState expects specific string literals or can be adapted for BaseMessages.
                    # Current agent.py uses tuples like ("user", content) or ("assistant", content)
                    # and also special ones like ("assistant_clarification", content), ("user_clarification_response", content)
                    # The generic "user" and "assistant" roles from Streamlit are fine here.
                    actual_role = "user" if role == "user" else "assistant" # Keep it simple
                    temp_history_for_agent.append((actual_role, msg_obj["content"]))
            
            current_graph_state_for_input = {
                "input_query": prompt, 
                "chat_history": temp_history_for_agent, 
                "reasoning_steps": [] # Start with fresh reasoning steps for a new query flow
            }
            # Clear any stale interrupt state if this is a truly new query flow
            if st.session_state.graph_config:
                 st.session_state.graph_config = None

        # Display reasoning steps in an expander
        # This was moved above the 'with st.chat_message("assistant"):' block.

        current_state_output = None
        graph_ended_naturally = False
        
        try:
            for event_output in app_graph.stream(current_graph_state_for_input, stream_mode="values"):
                current_state_output = event_output # This is the full AgentState dictionary

                # Update reasoning display
                reasoning_md_parts = []
                if current_state_output.get("current_node"):
                    reasoning_md_parts.append(f"**Current Node:** `{current_state_output['current_node']}`")
                
                if current_state_output.get("reasoning_steps"):
                    steps_str = "\n".join([f"- {s}" for s in current_state_output["reasoning_steps"]])
                    reasoning_md_parts.append(f"**Reasoning Log:**\n{steps_str}")
                
                if reasoning_md_parts:
                    reasoning_placeholder.markdown("\n\n".join(reasoning_md_parts))
                else:
                    reasoning_placeholder.markdown("Processing...")

                # Check if human feedback is needed
                if current_state_output.get("human_feedback_needed"):
                    st.session_state.graph_config = current_state_output # Save the entire state
                    clarification_q_raw = current_state_output.get("clarification_question", "I need more information. Could you please clarify?")
                    full_response = extract_content_from_message(clarification_q_raw)
                    response_placeholder.markdown(full_response)
                    # Crucially, ensure graph_config is set *before* breaking for interruption.
                    break # Stop graph execution, wait for human input via chat_input

                # Check for final response
                if current_state_output.get("llm_response"):
                    llm_response_raw = current_state_output["llm_response"]
                    full_response = extract_content_from_message(llm_response_raw)
                    response_placeholder.markdown(full_response)
                    # If an llm_response is produced, it implies this might be the end of the turn's output
                    # or a significant update. The loop continues until END or interruption.

            # After the loop, check how it finished.
            # If human_feedback_needed was set in the last state, the loop broke, and we handle it post-loop.
            if not (current_state_output and current_state_output.get("human_feedback_needed")):
                # Graph finished without requesting human input (i.e., reached an END node)
                graph_ended_naturally = True
                st.session_state.graph_config = None # Reset, turn is complete
                if current_state_output and current_state_output.get("llm_response"):
                        llm_response_raw = current_state_output.get("llm_response") # ensure latest is used
                        full_response = extract_content_from_message(llm_response_raw)
                        response_placeholder.markdown(full_response)
                elif not full_response and current_state_output: # Graph ended, no LLM response, but there was a state
                    # This case might occur if a node leads to END without setting llm_response (e.g., internal ops)
                    # However, all our ENDing nodes (greeting, unrelated, composite) *should* set llm_response.
                    # This is a fallback.
                    if current_state_output.get("status_message"): # Check if agent set a specific status
                         full_response = current_state_output.get("status_message")
                    else:
                         full_response = "I've processed your request. Let me know if you need anything else!" # Generic completion
                    response_placeholder.markdown(full_response)
                elif not full_response: # Loop finished, no current_state_output (empty stream?), or no llm_response.
                    full_response = "I'm not sure how to respond to that. Can you try rephrasing?"
                    response_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            full_response = "Sorry, I encountered an error. Please try again."
            response_placeholder.markdown(full_response)
            st.session_state.graph_config = None # Reset graph config on error
            # Log traceback for server-side debugging
            import traceback
            traceback.print_exc()
            current_state_output = None # Ensure no stale state is used

        # Ensure full_response is a string before appending
        if not isinstance(full_response, str):
            full_response = str(full_response) # Fallback conversion
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
elif graph_image_bytes == "PLAYWRIGHT_PYTHON_MISSING":
    st.sidebar.warning("Playwright Python package not installed. Graph image cannot be shown. Please run: `pip install playwright`")
elif isinstance(graph_image_bytes, str) and "PLAYWRIGHT_SETUP_ISSUE" in graph_image_bytes:
    st.sidebar.error(f"Playwright setup issue: {graph_image_bytes.replace('PLAYWRIGHT_SETUP_ISSUE: ', '')} Graph image cannot be shown. Please run `playwright install` and ensure browser binaries are available.")
elif isinstance(graph_image_bytes, str): # Other error message
    st.sidebar.error(f"Could not generate graph image: {graph_image_bytes}")
else: # Fallback if graph_image_bytes is None for some other reason
    st.sidebar.warning("Agent graph image is unavailable.")


st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangGraph & Streamlit.")

# TODO:
# - Add actual graph visualization if possible (e.g., using st.graphviz_chart if a DOT representation can be obtained from LangGraph)
# - Refine chat history management for the agent if specific formats (like BaseMessage objects) are required.
# - Enhance display of search results (e.g., cards for cheeses) if the agent provides structured search results in the state. 