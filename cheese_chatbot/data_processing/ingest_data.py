import json
import os
import sys
from dotenv import load_dotenv

# Assuming these utils are in the same directory or the path is configured
from embedding_utils import get_openai_embeddings, generate_embeddings
from mongo_utils import get_mongo_client, get_mongo_collection, insert_data_to_mongo, ensure_text_indexes
from pinecone_utils import get_pinecone_index, upsert_to_pinecone

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Define the path to the data file relative to the project root.
# This assumes 'ingest_data.py' is in 'cheese_chatbot/data_processing/'
# and 'data.json' is at the workspace root.
# You might need to adjust this path based on your actual project structure.
# For this script, we'll assume data.json is in the parent directory of 'cheese_chatbot'
# or adjust DATA_FILE_PATH as needed.
# If your data.json is at the same level as the cheese_chatbot folder:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE_PATH = os.path.join(PROJECT_ROOT, "data.json")
# If data.json is inside cheese_chatbot, then:
# DATA_FILE_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "data.json")

# Check if data.json exists where expected.
# The user has 'data.json' attached in their workspace root.
# So, the path should be relative to the workspace root.
# Given the script is at cheese_chatbot/data_processing/ingest_data.py,
# data.json is at ../../data.json from the script's location.
# However, the user provided the file as 'data.json' which implies it's in the current working directory
# when the script is run, or the path should be relative to the project root.
# Let's assume the user will run this script from the 'cheese_chatbot' directory,
# or the data.json is copied into 'cheese_chatbot' or 'cheese_chatbot/data_processing'.
# For robustness, let's try to locate it from the project root first (one level up from 'cheese_chatbot' folder)
# then try the workspace root as specified by the user.

# Let's use a path relative to where this script is, assuming data.json is at the root of the workspace
# containing the 'cheese_chatbot' folder.
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../cheese_chatbot/data_processing
CHEESE_CHATBOT_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # .../cheese_chatbot
WORKSPACE_ROOT = os.path.dirname(CHEESE_CHATBOT_DIR) # .../
ABSOLUTE_DATA_FILE_PATH = os.path.join(WORKSPACE_ROOT, "data.json")
print(f"ABSOLUTE_DATA_FILE_PATH: {ABSOLUTE_DATA_FILE_PATH}")

def load_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} items from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        raise

def preprocess_for_pinecone(cheese_data, embedding_model_instance):
    """
    Prepares data for Pinecone: generates embeddings and structures data.
    """

    # Helper function to convert complex metadata values to Pinecone-compatible types
    def stringify_metadata_value(value):
        if isinstance(value, dict):
            # Convert dict to "key1: value1, key2: value2, ..."
            return ", ".join(f"{str(k)}: {str(v)}" for k, v in value.items())
        elif isinstance(value, list):
            # Convert list to list of strings, filtering out None values
            return [str(x) for x in value if x is not None]
        # Return numbers, booleans, and simple strings as is
        return value

    pinecone_data = []
    texts_to_embed = []
    item_references = [] # To link back embeddings to items

    print("Preprocessing data for Pinecone...")
    for i, item in enumerate(cheese_data):
        # Concatenate relevant text fields for a richer embedding
        name = item.get("name", "")
        category = item.get("category", "")
        description = item.get("description", "")
        brand = item.get("brand","")
        warning_text = item.get("warning_text","")
        price = item.get("prices", "").get("Each", "")
        sku = item.get("sku", "")

        # Ensure all parts are strings
        text_for_embedding = f"Name: {name or 'Unknown'} Category: {category or 'Unknown'} Brand: {brand or 'Unknown'} Description: {description or 'No description'} Price: {price or 'No price'} Warning: {warning_text or 'No warning'} SKU: {sku or 'No SKU'}".strip()

        if not text_for_embedding or text_for_embedding == "Name: Unknown Category: Unknown Brand: Unknown Description: No description Price: No price Warning: No warning SKU: No SKU":
            print(f"Warning: Item SKU {item.get('sku', f'index_{i}')} has no text content for embedding. Skipping.")
            continue

        texts_to_embed.append(text_for_embedding)
        item_references.append(item)

    if not texts_to_embed:
        print("No valid items found to process for Pinecone after preprocessing.")
        return []

    print(f"Generating embeddings for {len(texts_to_embed)} items...")
    embeddings = generate_embeddings(texts_to_embed, embedding_model_instance)

    if not embeddings or len(embeddings) != len(item_references):
        print("Error: Embedding generation failed or mismatch in embedding count.")
        return []

    print("Successfully generated embeddings. Preparing data for Pinecone upsert...")
    for i, item in enumerate(item_references):
        sku = item.get("sku")
        if not sku:
            print(f"Warning: Item at original index (approx {cheese_data.index(item)}) is missing SKU. Using generated ID pinecone_item_{i}.")
            pinecone_id = f"pinecone_item_{i}"
        else:
            pinecone_id = str(sku)

        # Define the raw metadata payload
        raw_metadata = {
            "name": item.get("name", ""),
            "brand": item.get("brand", ""),
            "category": item.get("category", ""),
            "sku": str(item.get("sku", "")), 
            "price_per": item.get("pricePer", ""),
            # Extract specific 'Each' price as a float, handle potential errors
            "price": float(item.get("prices", {}).get("Each", "0").replace('$', '').strip() or "0"),
            "prices_details": item.get("prices", {}), # Keep as dict for stringify_metadata_value
            "weight_details": item.get("weights", {}),
            "dimensions_details": item.get("dimensions", {}), # This was the problematic field
            "itemCounts_details": item.get("itemCounts", {}),
            "warning": item.get("warning_text", ""),
            "description": item.get("description", ""),
            "href": item.get("href", ""),
            "priceOrder": item.get("priceOrder"), 
            "empty": item.get("empty"), 
            "related_skus": item.get("relateds", []), 
            "showImage": item.get("showImage")
        }

        # Process and filter metadata
        processed_metadata = {}
        for key, value in raw_metadata.items():
            if value is None: # Skip None values outright
                continue

            stringified_value = stringify_metadata_value(value)

            # Filter out None, empty strings, or empty lists after stringification
            if stringified_value is None:
                continue
            if isinstance(stringified_value, str) and not stringified_value.strip():
                continue
            if isinstance(stringified_value, list) and not stringified_value:
                continue
            
            processed_metadata[key] = stringified_value
        
        pinecone_data.append((pinecone_id, embeddings[i], processed_metadata))

    print(f"Prepared {len(pinecone_data)} items for Pinecone upsert.")
    return pinecone_data


def main():
    """
    Main function to orchestrate data loading, processing, and storage.
    """
    print(f"Starting data ingestion process. Attempting to load data from: {ABSOLUTE_DATA_FILE_PATH}")

    if not os.path.exists(ABSOLUTE_DATA_FILE_PATH):
        print(f"CRITICAL: data.json not found at the expected path: {ABSOLUTE_DATA_FILE_PATH}")
        print("Please ensure 'data.json' is in the root of your workspace (the directory containing the 'cheese_chatbot' folder).")
        return

    # 1. Load data from JSON file
    try:
        cheese_data_list = load_data(ABSOLUTE_DATA_FILE_PATH)
        if not cheese_data_list:
            print("No data loaded. Exiting.")
            return
    except Exception as e:
        print(f"Failed to load data: {e}. Exiting.")
        return # Exit if data loading fails

    mongo_client = None # Initialize to None for finally block

    try:
        # 2. Initialize MongoDB
        mongo_client = get_mongo_client()
        cheese_collection_mongo = get_mongo_collection(mongo_client)
        
        # Ensure MongoDB text indexes are created *before* inserting data,
        # or at least before any text search operations.
        print("Ensuring text indexes exist in MongoDB...")
        ensure_text_indexes(cheese_collection_mongo)

        # 3. Store raw data in MongoDB
        # We'll store the original items. If an item has no SKU, it will still be inserted.
        # MongoDB will generate an _id for it.
        # The `insert_data_to_mongo` utility already handles creating a unique index on 'sku'
        # and `ordered=False` to continue on duplicate errors.
        print("Storing data in MongoDB...")
        # Filter out "empty: true" items before MongoDB insertion as well, if desired.
        # Or let MongoDB store everything and filter during retrieval if needed.
        # For consistency with Pinecone, let's filter here.
        insert_data_to_mongo(cheese_collection_mongo, cheese_data_list)

        sys.exit()
        # 4. Initialize OpenAI Embeddings
        print("Initializing OpenAI embedding model...")
        embedding_model = get_openai_embeddings()

        # 5. Preprocess data and generate embeddings for Pinecone
        pinecone_upsert_data = preprocess_for_pinecone(cheese_data_list, embedding_model)

        if not pinecone_upsert_data:
            print("No data prepared for Pinecone. Skipping Pinecone operations.")
        else:
            # 6. Initialize Pinecone Index
            print("Initializing Pinecone index...")
            # Dimension for text-embedding-ada-002 is 1536
            pinecone_index = get_pinecone_index(dimension=1536)

            # 7. Upsert data to Pinecone
            if pinecone_index:
                upsert_to_pinecone(pinecone_index, pinecone_upsert_data)
            else:
                print("Pinecone index could not be initialized. Skipping upsert.")

        print("Data ingestion process completed.")

    except ValueError as ve: # Catching config errors from utils
        print(f"Configuration error: {ve}")
    except Exception as e:
        print(f"An error occurred during the ingestion process: {e}")
    finally:
        if mongo_client:
            mongo_client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    main()