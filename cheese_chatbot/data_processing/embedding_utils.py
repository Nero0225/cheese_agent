# Placeholder for embedding utility functions 

import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# It's good practice to keep API keys and other sensitive information
# in environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it up in your .env file.")

def get_openai_embeddings_model_name():
    return "text-embedding-ada-002"

def get_openai_embeddings(model_name="text-embedding-ada-002"):
    """
    Initializes and returns an OpenAIEmbeddings object.

    Args:
        model_name (str): The name of the OpenAI embedding model to use.
                          Defaults to "text-embedding-ada-002".

    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings class.
    """
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=model_name)
        return embeddings
    except Exception as e:
        print(f"Error initializing OpenAI embeddings: {e}")
        raise

def generate_embeddings(texts, embeddings_model):
    """
    Generates embeddings for a list of texts using the provided embeddings model.

    Args:
        texts (list[str]): A list of strings to embed.
        embeddings_model (OpenAIEmbeddings): An initialized OpenAIEmbeddings object.

    Returns:
        list[list[float]]: A list of embeddings, where each embedding is a list of floats.
                           Returns None if an error occurs.
    """
    if not texts:
        return []
    try:
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def generate_embedding_for_query(query, embeddings_model):
    """
    Generates an embedding for a single query string.

    Args:
        query (str): The query string to embed.
        embeddings_model (OpenAIEmbeddings): An initialized OpenAIEmbeddings object.

    Returns:
        list[float]: The embedding for the query.
                     Returns None if an error occurs.
    """
    if not query:
        return None
    try:
        return embeddings_model.embed_query(query)
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Ensure you have your OPENAI_API_KEY set in a .env file in the project root
    # or as an environment variable.
    print("Testing embedding utilities...")
    try:
        embedding_model_instance = get_openai_embeddings()
        sample_texts = [
            "Hello world",
            "This is a test of the OpenAI embedding system.",
            "Cheese is delicious."
        ]
        text_embeddings = generate_embeddings(sample_texts, embedding_model_instance)

        if text_embeddings:
            print(f"Successfully generated embeddings for {len(text_embeddings)} texts.")
            for i, embedding in enumerate(text_embeddings):
                print(f"Embedding for text {i+1} (first 5 dimensions): {embedding[:5]}...")
        else:
            print("Failed to generate text embeddings.")

        sample_query = "What kind of cheese is good for pizza?"
        query_embedding = generate_embedding_for_query(sample_query, embedding_model_instance)

        if query_embedding:
            print(f"\nSuccessfully generated embedding for query (first 5 dimensions): {query_embedding[:5]}...")
        else:
            print("Failed to generate query embedding.")

    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}") 