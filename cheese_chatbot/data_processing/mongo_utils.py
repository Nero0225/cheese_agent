# Placeholder for MongoDB utility functions 

import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage # For LLM interaction
import json # Added for parsing LLM JSON output

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "cheese_db") # Default DB name
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "cheeses") # Default collection

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables. Please set it up in your .env file.")

# Global client to be initialized by get_mongo_client
_mongo_client_instance = None

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
        raise
    except Exception as e:
        print(f"Error reading prompt file {file_path}: {e}")
        raise

def get_mongo_client():
    """
    Establishes a connection to MongoDB and returns the client.
    """
    global _mongo_client_instance
    if _mongo_client_instance is None:
        try:
            _mongo_client_instance = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # The ismaster command is cheap and does not require auth.
            # _mongo_client_instance.admin.command('ismaster')
            print("Successfully connected to MongoDB.")
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
            _mongo_client_instance = None # Ensure it remains None if connection fails
            raise
    return _mongo_client_instance

def get_mongo_collection(client, db_name=MONGO_DB_NAME, collection_name=MONGO_COLLECTION_NAME):
    """
    Returns a MongoDB collection object.
    """
    if not client:
        # Attempt to get a client if not provided, useful for standalone calls after app init
        client = get_mongo_client()
        if not client:
             raise ValueError("MongoDB client is not available.")
    try:
        db = client[db_name]
        collection = db[collection_name]
        return collection
    except Exception as e:
        print(f"Error getting MongoDB collection '{collection_name}' from database '{db_name}': {e}")
        raise

def insert_data_to_mongo(collection, data):
    """
    Inserts a list of documents into the specified MongoDB collection.
    Ensures 'sku' is unique if it exists in the documents.

    Args:
        collection: The MongoDB collection object.
        data (list[dict]): A list of dictionaries (documents) to insert.

    Returns:
        pymongo.results.InsertManyResult or None: Result of the insert operation,
                                                   or None if an error occurs or data is empty.
    """
    if not data:
        print("No data provided to insert.")
        return None
    if collection is None:
        raise ValueError("MongoDB collection is not initialized.")

    try:
        # Create a unique index on 'sku' if it doesn't exist.
        # This helps prevent duplicate entries based on SKU.
        if any("sku" in item for item in data):
            collection.create_index("sku", unique=True, background=True)

        # Consider using bulk operations for better performance with large datasets.
        # For now, insert_many is sufficient.
        result = collection.insert_many(data, ordered=False) # ordered=False continues on errors
        print(f"Successfully inserted {len(result.inserted_ids)} documents into MongoDB.")
        return result
    except OperationFailure as e:
        # Handle duplicate key errors specifically if 'sku' is meant to be unique
        if e.code == 11000: # Duplicate key error code
            print(f"Error inserting data due to duplicate key (likely SKU): {e.details}")
            # You might want to implement update logic here or skip duplicates
        else:
            print(f"MongoDB operation failed during insert: {e}")
        return None # Or re-raise specific errors if needed
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB insert: {e}")
        return None

def ensure_text_indexes(collection):
    """
    Ensures text indexes exist on specified fields for keyword search.
    Creates a compound text index.
    """
    if collection is None:
        raise ValueError("MongoDB collection is not initialized.")
    
    index_name = "text_search_index"
    text_index_fields = [
        ("name", "text"),
        ("category", "text"),
        ("description", "text"),
        ("brand", "text"),
        ("warning_text", "text")
    ]
    
    existing_indexes = collection.index_information()
    
    # Check if a text index with the desired fields already exists.
    # This check is a bit simplified. A more robust check would involve comparing
    # the 'key' field of existing_indexes for the specific compound text structure.
    already_exists = any(index_name in k or 
                         (existing_indexes[k].get('textIndexVersion') and 
                          all(tf in existing_indexes[k]['weights'] for tf, _ in text_index_fields))
                         for k in existing_indexes)

    if not already_exists:
        try:
            collection.create_index(text_index_fields, name=index_name, default_language='english')
            print(f"Text index '{index_name}' created successfully on fields: {[f[0] for f in text_index_fields]}.")
        except OperationFailure as e:
            print(f"Error creating text index '{index_name}': {e}")
            # Don't raise if index creation fails, but log it. Search might still work partially or fail.
    else:
        print(f"Text index '{index_name}' (or equivalent) already exists.")

def keyword_search_mongo(collection, user_query: str, llm_instance: any, limit: int = 10) -> list:
    """
    Performs a search in MongoDB using an LLM-generated query (filter or aggregation).
    Tries to infer if an aggregation query is needed, otherwise generates a filter query.

    Args:
        collection: The MongoDB collection object.
        user_query (str): The natural language user query.
        llm_instance (any): An initialized LLM instance capable of .invoke().
        limit (int): The maximum number of documents to return for filter queries.

    Returns:
        list: A list of documents from MongoDB, or an empty list if an error occurs
              or no results are found. For aggregation queries that return a single
              summary document (e.g. count), the list will contain that single document.
    """
    if collection is None:
        raise ValueError("MongoDB collection is not initialized.")
    if llm_instance is None:
        raise ValueError("LLM instance is required for LLM-generated MongoDB queries.")
    if not user_query:
        print("User query is empty. Returning empty list.")
        return []

    print(f"Original user query for MongoDB: '{user_query}'")
    results = []
    query_type_from_llm = "FILTER" # Default in case of issues

    try:
        prompt_template = load_prompt_from_file("mongo_unified_query_generation_prompt.txt")
        llm_prompt = prompt_template.format(query=user_query)
        
        response = llm_instance.invoke([HumanMessage(content=llm_prompt)])
        # print(response.content)
        generated_json_str = response.content.strip("```json").strip()
        print(f"LLM generated JSON response: {generated_json_str}")
        # Remove markdown ```json ... ``` if present
        # if generated_json_str.startswith("```json"):
        #     generated_json_str = generated_json_str[7:-3].strip()
        # elif generated_json_str.startswith("```"):
        #     generated_json_str = generated_json_str[3:-3].strip()
            
        # print(f"LLM generated JSON response: {generated_json_str}")

        llm_response_data = json.loads(generated_json_str)
        query_type_from_llm = llm_response_data.get("query_type", "FILTER").upper()
        query_body = llm_response_data.get("query_body")

        if not query_body:
            raise ValueError("LLM response missing 'query_body'.")

        print(f"LLM classified query as: {query_type_from_llm}, Query body: {query_body}")

        if query_type_from_llm == "AGGREGATION":
            if not isinstance(query_body, list):
                raise ValueError("Aggregation query_body must be a list.")
            if not query_body: # Empty pipeline by choice from LLM
                print("LLM returned an empty aggregation pipeline. No results.")
                results = []
            else:
                results = list(collection.aggregate(query_body))
                print(f"MongoDB aggregation query executed. Found {len(results)} results.")
        elif query_type_from_llm == "FILTER":
            if not isinstance(query_body, dict):
                raise ValueError("Filter query_body must be a dictionary.")
            projection = {'_id': 1, 'name': 1, 'sku': 1, 'brand': 1, 'category': 1, 'description': 1, 'prices': 1, 'images':1, 'href': 1, 'pricePer': 1, 'warning_text': 1, 'characteristics':1}
            results = list(collection.find(query_body, projection))
            print(f"MongoDB filter query executed. Found {len(results)} results.")
        else:
            print(f"Unknown query_type from LLM: '{query_type_from_llm}'. Defaulting to no results from LLM stage.")
            results = [] # Treat unknown type as no result from this stage

    except FileNotFoundError:
        print("Unified query generation prompt file not found. Cannot proceed with LLM query generation.")
        # No results from LLM stage, will go to fallback if `results` remains empty.
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM for unified query: {e}")
        print(f"LLM output was: {generated_json_str}")
        # No results from LLM stage
    except ValueError as e:
        print(f"Error in LLM response structure or query body: {e}")
        # No results from LLM stage
    except OperationFailure as e:
        print(f"MongoDB operation failed for LLM-generated query: {e}")
        # No results from LLM stage
    except Exception as e:
        print(f"An unexpected error occurred during LLM query generation/execution: {e}")
        # No results from LLM stage
    
    if not results:
        print(f"LLM-generated query ('{query_type_from_llm}' type attempt) failed or returned no results. Falling back to basic text search for: '{user_query}'")
        try:
            ensure_text_indexes(collection) # Ensure text index exists
            results = list(collection.find(
                {'$text': {'$search': user_query}},
                {'score': {'$meta': 'textScore'}, '_id': 1, 'name': 1, 'sku': 1, 'brand': 1, 'category': 1, 'description': 1, 'prices': 1, 'images':1, 'href': 1}
            ).sort([('score', {'$meta': 'textScore'})])) #.limit(limit))
            print(f"Fallback MongoDB text search found {len(results)} results for query '{user_query}'.")
        except OperationFailure as e:
            print(f"Fallback MongoDB text search failed: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during fallback MongoDB text search: {e}")
            return []

    # Convert ObjectId to string for JSON serialization if needed later
    for item in results:
        if '_id' in item and not isinstance(item['_id'], str): # Check if not already string (e.g. from aggregation)
            item['_id'] = str(item['_id'])
    
    return results