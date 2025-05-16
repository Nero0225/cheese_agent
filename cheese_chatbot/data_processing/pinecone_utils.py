# Placeholder for Pinecone utility functions 

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cheese-index") # Default index name

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set them up in your .env file.")

def init_pinecone():
    """
    Initializes Pinecone connection.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("Successfully initialized Pinecone.")
        return pc
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise

def get_pinecone_index(index_name=PINECONE_INDEX_NAME, dimension=1536, metric='cosine'):
    """
    Gets a Pinecone index. Creates it if it doesn't exist.

    Args:
        index_name (str): The name of the Pinecone index.
        dimension (int): The dimension of the vectors to be stored.
                         For OpenAI's text-embedding-ada-002, this is 1536.
        metric (str): The distance metric to use (e.g., 'cosine', 'euclidean', 'dotproduct').

    Returns:
        pinecone.Index: An instance of the Pinecone index.
    """
    pc = init_pinecone() # Ensure Pinecone is initialized
    try:
        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' not found. Creating new index with dimension {dimension} and metric '{metric}'...")
            pc.create_index(index_name, dimension=dimension, metric=metric, spec=ServerlessSpec(cloud='aws', region='us-east-1')) # or 's1' depending on your needs
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")
        index = pc.Index(index_name)
        return index
    except Exception as e:
        print(f"Error getting or creating Pinecone index '{index_name}': {e}")
        raise

def upsert_to_pinecone(index, vectors_with_metadata, batch_size=100):
    """
    Upserts vectors with metadata to the specified Pinecone index.

    Args:
        index (pinecone.Index): The Pinecone index object.
        vectors_with_metadata (list[tuple]): A list of tuples, where each tuple is
                                             (id, vector, metadata_dict).
                                             'id' should be a unique string for each vector.
                                             'vector' is the embedding list of floats.
                                             'metadata_dict' is a dictionary of metadata associated with the vector.
        batch_size (int): The number of vectors to upsert in each batch.

    Returns:
        pinecone.UpsertResponse or None: Response from Pinecone or None if error.
    """
    if not vectors_with_metadata:
        print("No data provided for Pinecone upsert.")
        return None
    if not index:
        raise ValueError("Pinecone index is not initialized.")

    try:
        print(f"Starting upsert of {len(vectors_with_metadata)} vectors to Pinecone index ...")
        total_upserted = 0
        for i in range(0, len(vectors_with_metadata), batch_size):
            batch = vectors_with_metadata[i:i + batch_size]
            response = index.upsert(vectors=batch)
            total_upserted += response.upserted_count
            print(f"Upserted batch {i//batch_size + 1}, {response.upserted_count} vectors.")
        print(f"Successfully upserted a total of {total_upserted} vectors to Pinecone.")
        # The response object for the last batch is returned, which might not be ideal.
        # For simplicity, we're just confirming the operation.
        # You might want to aggregate responses or handle them differently.
        return response # Returns the response of the last batch
    except Exception as e:
        print(f"An unexpected error occurred during Pinecone upsert: {e}")
        return None

def query_pinecone(index, vector, top_k=5, filter_metadata=None):
    """
    Queries the Pinecone index.

    Args:
        index (pinecone.Index): The Pinecone index object.
        vector (list[float]): The query vector.
        top_k (int): The number of results to return.
        filter_metadata (dict, optional): Metadata filter to apply. Defaults to None.

    Returns:
        pinecone.QueryResponse or None: Query results or None if an error.
    """
    if not index:
        raise ValueError("Pinecone index is not initialized.")
    if not vector:
        print("No query vector provided.")
        return None

    try:
        if filter_metadata:
            results = index.query(vector=vector, top_k=top_k, filter=filter_metadata, include_metadata=True)
        else:
            results = index.query(vector=vector, top_k=top_k, include_metadata=True)
        # print(f"Pinecone query returned {len(results.matches)} matches.")
        return results
    except Exception as e:
        print(f"An unexpected error occurred during Pinecone query: {e}")
        return None


if __name__ == '__main__':
    # Example Usage
    # Ensure PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME are set in .env
    # Also, ensure your OpenAI API key is set for the embedding_utils part.
    print("Testing Pinecone utilities...")
    try:
        # This import is here for the __main__ example, normally it would be at the top
        from embedding_utils import get_openai_embeddings, generate_embeddings, generate_embedding_for_query

        pinecone_index = get_pinecone_index()
        print(f"Pinecone index stats: {pinecone_index.describe_index_stats()}")

        # Sample data for upsertion
        # (id, vector, metadata_dict)
        # In a real scenario, vectors would come from your embedding model
        embedding_model_instance = get_openai_embeddings()
        sample_texts_for_pinecone = [
            {"id": "pc_test_001", "text": "A creamy brie cheese.", "metadata": {"category": "Soft Cheese", "brand": "President"}},
            {"id": "pc_test_002", "text": "A sharp cheddar.", "metadata": {"category": "Hard Cheese", "brand": "Tillamook"}},
            {"id": "pc_test_003", "text": "Gorgonzola blue cheese.", "metadata": {"category": "Blue Cheese", "brand": "Galbani"}}
        ]

        vectors_to_upsert = []
        for item in sample_texts_for_pinecone:
            embedding = generate_embedding_for_query(item["text"], embedding_model_instance)
            if embedding:
                vectors_to_upsert.append((item["id"], embedding, item["metadata"]))

        if vectors_to_upsert:
            # Optional: Delete test items if they exist to ensure clean run
            # print("Deleting test items from Pinecone if they exist...")
            # pinecone_index.delete(ids=[item["id"] for item in sample_texts_for_pinecone])
            # import time
            # time.sleep(5) # Give some time for deletes to process

            upsert_response = upsert_to_pinecone(pinecone_index, vectors_to_upsert)
            if upsert_response:
                print(f"Pinecone upsert response (last batch): {upsert_response}")
                # It might take a moment for upserts to be reflected in stats
                # import time
                # time.sleep(10) # Wait for index to update
                print(f"Pinecone index stats after upsert: {pinecone_index.describe_index_stats()}")
        else:
            print("Could not generate embeddings for sample texts.")


        # Sample query
        if vectors_to_upsert: # Only query if we upserted something
            query_text = "What is a good soft cheese?"
            query_embedding = generate_embedding_for_query(query_text, embedding_model_instance)

            if query_embedding:
                print(f"\nQuerying Pinecone for: '{query_text}'")
                query_results = query_pinecone(pinecone_index, query_embedding, top_k=2)
                if query_results and query_results.matches:
                    print("Query results:")
                    for match in query_results.matches:
                        print(f"  ID: {match.id}, Score: {match.score:.4f}, Metadata: {match.metadata}")
                else:
                    print("No results found or error in query.")
            else:
                print("Could not generate query embedding.")

        # Optional: Clean up by deleting the test index if you created it for this test
        # print(f"\nConsider deleting the test index '{PINECONE_INDEX_NAME}' if it was created for this test.")
        # pinecone.delete_index(PINECONE_INDEX_NAME)
        # print(f"Index '{PINECONE_INDEX_NAME}' deleted.")

    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during Pinecone testing: {e}") 