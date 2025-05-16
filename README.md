# 🧀 Kimelo's Cheese Chatbot

## Overview

Kimelo's Cheese Chatbot is an AI-powered conversational agent designed to assist users in finding information about various cheeses. It leverages Large Language Models (LLMs) and a knowledge base to understand user queries, perform searches, and generate informative responses. The chatbot can handle cheese-related questions, provide recommendations, and engage in conversational greetings. It also features a human-in-the-loop capability for clarifying ambiguous queries.

This project demonstrates the use of LangGraph to build a robust and stateful agent, with integrations for data retrieval from MongoDB and Pinecone, and a Streamlit application for user interaction.

## Features

-   **Conversational AI**: Engages users with natural language understanding and generation.
-   **Cheese Knowledge Base**: Accesses information about a variety of cheeses (details, pricing, categories, etc.).
-   **Dual Search Strategy**:
    -   **MongoDB Keyword Search**: For targeted searches based on keywords.
    -   **Pinecone Semantic Search**: For vector-based similarity searches on user queries.
    -   LLM-driven decision on which strategy to use per query.
-   **Query Classification**: Identifies user intent (e.g., greeting, cheese query, unrelated).
-   **Human-in-the-Loop**: If a query is ambiguous or search results are unclear, the agent can ask the user for clarification.
-   **Reasoning Transparency**: Displays the agent's internal reasoning steps and current node in the Streamlit UI.
-   **User Interfaces**:
    -   Interactive Streamlit web application.
    -   Console-based application for testing and development.
-   **Dynamic Prompting**: Utilizes external prompt files for easy modification and management of LLM instructions.

## Tech Stack

-   **Programming Language**: Python
-   **Agent Framework**: LangGraph
-   **LLMs**: OpenAI (e.g., GPT-4o)
-   **Vector Database**: Pinecone (for semantic search)
-   **NoSQL Database**: MongoDB (for keyword-based search and data storage)
-   **Web Application**: Streamlit
-   **Environment Management**: `dotenv`
-   **Embeddings**: OpenAI Embeddings

## Project Structure

```
cheese_chatbot/
├── agent/                  # Core agent logic, nodes, and state definition
│   ├── __init__.py
│   ├── agent.py            # Main agent graph definition and node functions
│   └── state.py            # Defines the AgentState for the graph
├── data_processing/        # Utilities for interacting with data sources
│   ├── __init__.py
│   ├── embedding_utils.py  # Embedding model utilities
│   ├── mongo_utils.py      # MongoDB connection and search functions
│   └── pinecone_utils.py   # Pinecone connection and search functions
├── prompts/                # Prompt templates for various agent tasks
│   ├── classify_query_prompt.txt
│   ├── decide_search_strategy_prompt.txt
│   ├── evaluate_search_clarification_prompt.txt
│   └── generate_composite_response_prompt.txt
├── static/                 # Static assets for the Streamlit app
│   ├── __init__.py
│   └── graph.png           # Static image of the agent's graph structure
├── __init__.py
├── app.py                  # Streamlit web application
├── console_test_app.py     # Console-based test application
└── requirements.txt        # Python dependencies

data.json                   # Root level: Sample cheese data (used for populating databases)
.env                        # Root level: Environment variables (not committed)
README.md                   # This file
```

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)
-   Access to OpenAI, MongoDB, and Pinecone services.

### Installation Steps

1.  **Clone the Repository** (if applicable)
    ```bash
    # git clone <repository_url>
    # cd cheese_chatbot_project
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory of the project and add your API keys and service URIs:
    ```env
    OPENAI_API_KEY="your_openai_api_key"

    # MongoDB Configuration
    MONGO_URI="your_mongodb_connection_string"
    MONGO_DATABASE_NAME="your_mongodb_database_name" # e.g., "cheese_shop"
    MONGO_COLLECTION_NAME="your_mongodb_collection_name" # e.g., "cheeses"

    # Pinecone Configuration
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIRONMENT="your_pinecone_environment" # e.g., "gcp-starter" or "us-west1-gcp" etc.
    PINECONE_INDEX_NAME="your_pinecone_index_name" # e.g., "cheese-index"

    # Optional: LangSmith for tracing (if you uncommented it in app.py)
    # LANGCHAIN_TRACING_V2="true"
    # LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    # LANGCHAIN_API_KEY="your_langchain_api_key"
    # LANGCHAIN_PROJECT="your_langchain_project_name" # e.g., cheese-chatbot-dev
    ```

5.  **Data Population**:
    -   The `data.json` file at the root contains sample cheese data.
    -   **MongoDB**: The `mongo_utils.py` script includes a function (`get_mongo_collection`) that, if the specified collection is empty, attempts to populate it from `data.json`. Ensure `data.json` is in the project root when running for the first time if the collection is new/empty.
    -   **Pinecone**: The `pinecone_utils.py` script includes a function (`get_pinecone_index`) that, if the index is empty or has few vectors, attempts to populate it by embedding data from `data.json`. Make sure your Pinecone index is configured with the correct dimension for your chosen OpenAI embedding model (e.g., 1536 for `text-embedding-ada-002`).

6.  **Playwright for Graph Visualization (if dynamic generation is re-enabled)**
    The Streamlit app (`app.py`) is currently configured to load a static graph image from `cheese_chatbot/static/graph.png`. If you wish to re-enable dynamic graph image generation in `app.py` (which uses `draw_mermaid_png()`), you'll need Playwright:
    ```bash
    pip install playwright
    playwright install
    ```

## Running the Application

### Streamlit Web App

Navigate to the project's root directory and run:
```bash
streamlit run cheese_chatbot/app.py
```
This will start the interactive web application. The sidebar displays a static image of the agent's graph structure.

### Console Test App

For a command-line interface:
```bash
python cheese_chatbot/console_test_app.py
```

## Agent Architecture

The core of the chatbot is an agent built with LangGraph. The agent follows a cyclical graph structure:

1.  **`classify_query`**: The user's input is classified into intents like "greeting", "cheese_query", or "unrelated".
2.  **Branching based on Intent**:
    -   Greetings go to `generate_greeting_response`.
    -   Unrelated queries go to `handle_unrelated_query`.
    -   Cheese queries proceed to `search_cheese`.
3.  **`search_cheese`**: An LLM decides whether to use MongoDB (keyword search) or Pinecone (semantic search) based on the query. The chosen search is executed.
4.  **`evaluate_search`**: The search results are evaluated by an LLM. It determines if the results are sufficient to answer the query or if clarification from the user is needed.
5.  **Conditional Branching**:
    -   If clarification is needed (`human_feedback_needed` is true), the agent goes to `request_human_input`.
    -   Otherwise, it proceeds to `generate_composite_response`.
6.  **`request_human_input`**: The graph interrupts, and the Streamlit/console app displays the clarification question to the user.
7.  **`process_human_input`**: The user's clarifying response is taken as the new input query, and the flow returns to `classify_query`.
8.  **`generate_composite_response`**: An LLM generates a final response to the user based on the original query and the retrieved search results.
9.  **END**: Nodes like `generate_greeting_response`, `handle_unrelated_query`, and `generate_composite_response` lead to the end of the current interaction cycle.

## Prompts

The `cheese_chatbot/prompts/` directory contains text files with prompt templates used by the LLMs for various tasks (classification, search strategy decision, evaluation, response generation). This allows for easy modification and tuning of the agent's behavior without altering the core Python code.

## Future Enhancements

-   Implement more sophisticated data ingestion and synchronization pipelines.
-   Expand the range of tools available to the agent (e.g., live web search for cheese news).
-   Add user authentication and personalized chat history.
-   Develop more advanced error handling and recovery mechanisms within the agent.
-   Integrate a dynamic graph visualization in Streamlit if performance allows.
-   Offer more interactive ways to explore search results in the UI. 