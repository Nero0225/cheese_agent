# cheese_chatbot/console_test_app.py
import os
from dotenv import load_dotenv
from langgraph.types import Command
# Attempt to import agent components. User needs to ensure these are correctly set up.
try:
    from agent.agent import get_graph, get_tools_resources
except ImportError as e:
    print("Error: Could not import agent components from agent.agent.")
    print(f"Detailed error: {e}")
    import traceback
    traceback.print_exc()
    print("Please ensure agent/agent.py and its dependencies are correctly set up.")
    exit(1)

# --- Helper Function to Extract Content (similar to app.py) ---
def extract_content_from_message(message_output):
    """Extracts string content from various message types."""
    if isinstance(message_output, str):
        return message_output
    # Check if it's an AIMessage or similar object with a 'content' attribute
    if hasattr(message_output, 'content') and isinstance(getattr(message_output, 'content'), str):
        return getattr(message_output, 'content')
    # Fallback for other types or if content is not a string
    print(f"[Debug] extract_content_from_message: input type {type(message_output)}, converting to str.")
    return str(message_output)

# --- Agent Initialization ---
def initialize_agent_console():
    """Initializes the agent graph and necessary resources for console app."""
    print("Initializing agent...")
    # Load environment variables from .env file
    load_dotenv()
    try:
        resources = get_tools_resources()
        graph = get_graph(resources)
        print("Agent initialized successfully.")
        return graph
    except Exception as e:
        print(f"Error during agent initialization: {e}")
        print("Please check your environment variables and agent/resource setup.")
        exit(1)

# --- Main Console Chat Logic ---
def run_console_chat():
    """Runs the console-based chat application."""
    print("\n🧀 Kimelo's Cheese Chatbot (Console Version) 🧀")
    print("Type 'quit' to exit.")
    print("-" * 40)

    app_graph_console = initialize_agent_console()

    # Initialize chat history
    chat_history_console = [{"role": "assistant", "content": "Hello! How can I help you find the perfect cheese today?"}]
    print(f"Assistant: {chat_history_console[0]['content']}")

    current_graph_state_console = None # To store graph state during human-in-the-loop

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() == 'quit':
            print("Assistant: Goodbye! Feel free to chat again anytime.")
            break

        chat_history_console.append({"role": "user", "content": prompt})
        
        inputs = {}
        # Check if we are responding to a clarification request from the agent
        if current_graph_state_console and current_graph_state_console.get("current_node") == "request_human_input":
            inputs = current_graph_state_console.copy()
            inputs["human_response"] = prompt # User's new input is the human_response.
            # Clear the flag indicating we were waiting, process_human_input will take over
            current_graph_state_console = None 
            Command(resume={"human_response": prompt})
        else:
            chat_history_for_agent = []
            if len(chat_history_console) > 1:
                history_tuples = []
                for msg_obj in chat_history_console[:-1]:
                    role = msg_obj["role"]
                    history_tuples.append((role, msg_obj["content"]))
                chat_history_for_agent = history_tuples
            
            inputs = {
                "input_query": prompt, 
                "chat_history": chat_history_for_agent, 
                "reasoning_steps": [],
                "original_query_for_clarification": prompt # For the first turn, this is the same
            }
            if current_graph_state_console: # Clear any stale interrupt state
                 current_graph_state_console = None

        print("\n💡 Reasoning Process:")
        full_response_console = ""
        final_assistant_message_for_history = ""
        current_state_output_for_loop = None

        for event_output in app_graph_console.stream(inputs, stream_mode="values"):
            
            try:
                if event_output['__interrupt__']:
                    current_state_output_for_loop = event_output['__interrupt__'][0].value['state']
                print("Debugging console_test_app.py line 102: current_state_output_for_loop: ", current_state_output_for_loop.get("current_node"))
            except:
                current_state_output_for_loop = event_output
    
            reasoning_md_parts = []
            if current_state_output_for_loop.get("current_node"):
                reasoning_md_parts.append(f"  Current Node: {current_state_output_for_loop['current_node']}")
            
            if current_state_output_for_loop.get("reasoning_steps"):
                steps = current_state_output_for_loop["reasoning_steps"]
                if isinstance(steps, list):
                    # Display only new reasoning steps for this stream event if possible (tricky with full state stream)
                    # For simplicity, here we might re-print or show last N steps.
                    # For now, printing the tail if it grows too long for console
                    display_steps = steps[-5:] if len(steps) > 5 else steps 
                    steps_str = "\n".join([f"  - {str(s)}" for s in display_steps])
                    if len(steps) > 5:
                        steps_str = f"  ... (last 5 of {len(steps)} steps)\n{steps_str}"
                    reasoning_md_parts.append(f"  Reasoning Log:\n{steps_str}")
                else:
                    reasoning_md_parts.append(f"  Reasoning Log (raw): {steps}")
            
            # if reasoning_md_parts:
            #     print("\n".join(reasoning_md_parts))
            # else:
            #     print("  Processing...")

            # Check if the agent is now waiting for human input
            if current_state_output_for_loop.get("current_node") == "request_human_input":
                current_graph_state_console = current_state_output_for_loop.copy()
                # The request_human_input_node itself prints "AGENT ASKS: ..."
                # and sets status_message. We use status_message for the console output.
                question_to_ask = current_state_output_for_loop.get("clarification_question", "I need more information. Could you please clarify?")
                # The actual print of "AGENT ASKS:" is done by the node. 
                # We just need to inform the user that a response is awaited based on what the node said.
                # However, the node's print might not be visible if stream is handled differently or in a UI.
                # So, printing the question here from status_message ensures visibility.
                print(f"\nAssistant: {question_to_ask}") # Use status_message
                final_assistant_message_for_history = question_to_ask # Save for history
                break # Stop graph execution, wait for human input for the next loop iteration

            if current_state_output_for_loop.get("llm_response"):
                final_assistant_message_for_history = extract_content_from_message(current_state_output_for_loop["llm_response"])

        # After the loop finishes or breaks due to needing human input
        if current_state_output_for_loop:
            if current_state_output_for_loop.get("current_node") == "request_human_input":
                # Already handled printing the question and saving state, just need to ensure history is updated.
                full_response_console = final_assistant_message_for_history
            elif final_assistant_message_for_history: # LLM Response was set
                full_response_console = final_assistant_message_for_history
                print(f"\nAssistant: {full_response_console}")
            elif not final_assistant_message_for_history: # Graph ended with no llm_response and not asking for input
                full_response_console = "I've completed my current task. Do you have any other cheese questions?"
                print(f"\nAssistant: {full_response_console}")
        else: # Should not happen if graph streams at least one state
            full_response_console = "Sorry, an unexpected issue occurred."
            print(f"\nAssistant: {full_response_console}")
        
        if not isinstance(full_response_console, str):
            full_response_console = str(full_response_console)
        
        if full_response_console: # Avoid adding empty messages to history
            chat_history_console.append({"role": "assistant", "content": full_response_console})
        
        print("-" * 140)

if __name__ == "__main__":
    run_console_chat() 