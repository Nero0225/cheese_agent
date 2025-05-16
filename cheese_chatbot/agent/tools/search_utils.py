# Utility functions for search, including hybrid search

from typing import List, Dict, Any, Set
from data_processing.mongo_utils import keyword_search_mongo
# We expect embedding_model to be an instance of something like OpenAIEmbeddings
# and pinecone_index to be a Pinecone index object.

def hybrid_cheese_search(
    query: str, 
    mongo_collection: Any, # pymongo.collection.Collection
    pinecone_index: Any,   # pinecone.Index
    embedding_model: Any,  # e.g., langchain_openai.OpenAIEmbeddings
    llm_instance: Any,     # LLM instance for keyword refinement in MongoDB search
    top_k_semantic: int = 5, 
    top_k_keyword: int = 5,
    semantic_weight: float = 0.7, # Placeholder for future re-ranking, not used yet
    keyword_weight: float = 0.3   # Placeholder for future re-ranking, not used yet
) -> List[Dict[str, Any]]:
    """
    Performs a hybrid search for cheeses using both semantic (Pinecone) and keyword (MongoDB) search.
    Combines and de-duplicates the results.
    """
    print(f"Hybrid search initiated for query: '{query}'")
    reasoning_steps_search = [] # For local reasoning, not directly part of AgentState here

    # 1. Semantic Search (Pinecone)
    semantic_results_processed: List[Dict[str, Any]] = []
    try:
        query_vector = embedding_model.embed_query(query)
        reasoning_steps_search.append(f"Generated query vector for semantic search.")
        
        pinecone_response = pinecone_index.query(
            vector=query_vector, 
            top_k=top_k_semantic, 
            include_metadata=True
        )
        reasoning_steps_search.append(f"Pinecone query returned {len(pinecone_response.get('matches', []))} matches.")
        
        for match in pinecone_response.get('matches', []):
            # The ID in Pinecone was set to be the SKU during ingestion.
            # Metadata should contain the original cheese data.
            if match.get('metadata'):
                # Ensure essential fields are present, especially SKU
                # Pinecone ID is the SKU
                cheese_data = dict(match.get('metadata'))
                cheese_data['sku'] = match.get('id', cheese_data.get('sku')) # Ensure SKU is present
                cheese_data['search_score_semantic'] = match.get('score', 0.0) # Store relevance score
                # Remove '_id' if it came from metadata to avoid confusion with mongo's _id
                if '_id' in cheese_data: # it shouldn't be if ingested correctly
                    del cheese_data['_id']
                semantic_results_processed.append(cheese_data)
            else:
                reasoning_steps_search.append(f"Warning: Pinecone match ID {match.get('id')} missing metadata.")
        reasoning_steps_search.append(f"Processed {len(semantic_results_processed)} semantic results from Pinecone.")

    except Exception as e:
        print(f"Error during semantic search: {e}")
        reasoning_steps_search.append(f"Semantic search failed: {e}")

    # 2. Keyword Search (MongoDB)
    keyword_results_processed: List[Dict[str, Any]] = []
    try:
        # keyword_search_mongo returns a list of dicts
        mongo_matches = keyword_search_mongo(mongo_collection, query, llm_instance, limit=top_k_keyword)
        reasoning_steps_search.append(f"MongoDB keyword search returned {len(mongo_matches)} matches.")
        for item in mongo_matches:
            # Convert ObjectId to string if present, and ensure SKU is available
            if '_id' in item:
                item['_id'] = str(item['_id']) 
            # SKU should already be a field. Add score if available (keyword_search_mongo includes 'score')
            item['search_score_keyword'] = item.get('score', 0.0) # textScore from MongoDB
            keyword_results_processed.append(item)
        reasoning_steps_search.append(f"Processed {len(keyword_results_processed)} keyword results from MongoDB.")

    except Exception as e:
        print(f"Error during keyword search: {e}")
        reasoning_steps_search.append(f"Keyword search failed: {e}")

    # 3. Combine and De-duplicate Results
    # Using SKU as the primary key for de-duplication.
    combined_results: Dict[str, Dict[str, Any]] = {}
    processed_skus: Set[str] = set()

    # Process semantic results first
    for item in semantic_results_processed:
        sku = item.get('sku')
        if sku and sku not in processed_skus:
            combined_results[sku] = item
            processed_skus.add(sku)
        elif sku and sku in processed_skus: # Item already added by semantic search, potentially update with keyword score
             if 'search_score_keyword' not in combined_results[sku] and 'search_score_keyword' in item: # Should not happen here
                combined_results[sku]['search_score_keyword'] = item['search_score_keyword']


    # Process keyword results, adding if new, or potentially merging scores/info
    for item in keyword_results_processed:
        sku = item.get('sku')
        if sku and sku not in processed_skus:
            combined_results[sku] = item
            processed_skus.add(sku)
        elif sku and sku in processed_skus: # Item already added by semantic search
            # Add keyword score if it's not there
            if 'search_score_keyword' not in combined_results[sku]:
                 combined_results[sku]['search_score_keyword'] = item.get('search_score_keyword', 0.0)
            # Optionally, update other fields if keyword result is richer, or combine descriptions.
            # For now, prioritize the one already there (from semantic) and just add the keyword score.

    final_deduplicated_list = list(combined_results.values())
    reasoning_steps_search.append(f"Combined and de-duplicated results. Total unique items: {len(final_deduplicated_list)}.")
    
    # 4. Optional: Re-ranking (simple sort by semantic score for now if available, then keyword score)
    # A more sophisticated re-ranking could use the weights or a more complex model.
    # For now, just sort by semantic then keyword score (descending for higher is better).
    final_deduplicated_list.sort(key=lambda x: (x.get('search_score_semantic', 0.0), x.get('search_score_keyword', 0.0)), reverse=True)
    reasoning_steps_search.append(f"Results sorted (primarily by semantic score, then keyword score).")

    print("\nHybrid Search Internal Reasoning:")
    for step in reasoning_steps_search:
        print(f"- {step}")
    
    # Ensure all items in the final list have the core fields expected by the agent.
    # The ingestion process should ensure 'name', 'sku', 'description', 'category', 'brand' are in metadata.
    # MongoDB results will have them directly.
    
    return final_deduplicated_list


if __name__ == '__main__':
    # This basic test requires setting up dummy/mock objects for DB connections
    # or actual connections with environment variables.
    print("Testing hybrid_cheese_search (basic placeholder test)...")
    
    # Mocking dependencies for a basic test run:
    class MockEmbeddingModel:
        def embed_query(self, text: str) -> List[float]:
            print(f"(MockEmbeddingModel) Embedding query: '{text}'")
            return [0.1] * 10 

    class MockPineconeMatch:
        def __init__(self, id: str, score: float, metadata: Dict[str, Any]):
            self.id = id
            self.score = score
            self.metadata = metadata
        
        def to_dict(self):
            return {'id': self.id, 'score': self.score, 'metadata': self.metadata}

    class MockPineconeResponse:
         def __init__(self, matches: List[MockPineconeMatch]):
             self.matches = matches
         
         def get(self, key, default=None):
             if key == 'matches':
                 return [m.to_dict() for m in self.matches]
             return default

    class MockPineconeIndex:
        def query(self, vector: List[float], top_k: int, include_metadata: bool) -> MockPineconeResponse:
            print(f"(MockPineconeIndex) Querying with vector (len {len(vector)}), top_k={top_k}, include_metadata={include_metadata}")
            matches = [
                MockPineconeMatch(id="S123", score=0.9, metadata={"name": "Semantic Brie", "sku": "S123", "category": "Soft", "description": "A creamy Brie."}),
                MockPineconeMatch(id="S456", score=0.85, metadata={"name": "Semantic Cheddar", "sku": "S456", "category": "Hard", "description": "A sharp Cheddar."})
            ]
            return MockPineconeResponse(matches=matches[:top_k])

    # MockMongoCollection needs to be updated to simulate find and aggregate based on LLM-like queries
    class MockMongoCollection:
        def __init__(self):
            self.data_store = {
                "S123": {"name": "Keyword Brie", "sku": "S123", "category": "Soft", "description": "A lovely Brie.", "brand": "French Cheeses Inc.", "characteristics": {"texture": "creamy", "milk_type": "cow"}, "_id": "mongo_id_2"},
                "K789": {"name": "Keyword Cheddar", "sku": "K789", "category": "Hard", "description": "A fine Cheddar.", "brand": "Old World Farm", "characteristics": {"texture": "firm", "milk_type": "cow"}, "_id": "mongo_id_1"},
                "G555": {"name": "Goat Gouda", "sku": "G555", "category": "Semi-hard", "description": "A mild goat gouda.", "brand": "Dutch Masters", "characteristics": {"texture": "firm", "milk_type": "goat"}, "_id": "mongo_id_3"}
            }

        def find(self, query_filter: Dict[str, Any], projection: Dict[str, Any] = None, limit: int = 10):
            print(f"(MockMongoCollection) Finding with LLM-generated filter: {query_filter}")
            # Simplified simulation of MongoDB find based on common LLM outputs
            # This mock won't fully replicate MongoDB's $regex or complex operators perfectly.
            results = []
            for doc in self.data_store.values():
                match = True
                if not query_filter: # Empty filter matches all
                    pass
                elif "$and" in query_filter:
                    for condition in query_filter["$and"]:
                        field, expr = list(condition.items())[0]
                        if isinstance(expr, dict) and "$regex" in expr:
                            # Basic regex check (case-insensitive substring)
                            if expr["$regex"].lower() not in doc.get(field, "").lower():
                                match = False
                                break
                        elif doc.get(field) != expr:
                            match = False
                            break
                    if not query_filter["$and"]: # empty $and also matches all
                        match = True 

                elif "characteristics.milk_type" in query_filter and "$regex" in query_filter["characteristics.milk_type"]:
                    if query_filter["characteristics.milk_type"]["$regex"].lower() not in doc.get("characteristics", {}).get("milk_type", "").lower():
                        match = False
                elif query_filter: # other simple field:value matches
                    for key, value in query_filter.items():
                        if isinstance(value, dict) and "$regex" in value:
                             if value["$regex"].lower() not in doc.get(key, "").lower():
                                match = False; break
                        elif doc.get(key) != value:
                            match = False; break
                
                if match:
                    # Simulate projection (crude)
                    projected_doc = {k: doc[k] for k in projection if k in doc} if projection else doc.copy()
                    projected_doc['score'] = 1.0 # Mock score for text search like behavior
                    results.append(projected_doc)
            
            return MockMongoCursor(results, limit_val=limit) # keyword_search_mongo handles limit now

        def aggregate(self, pipeline: List[Dict[str, Any]]):
            print(f"(MockMongoCollection) Aggregating with LLM-generated pipeline: {pipeline}")
            # Highly simplified aggregation mock for count
            if pipeline and len(pipeline) == 2 and "$match" in pipeline[0] and "$count" in pipeline[1]:
                match_filter = pipeline[0]["$match"]
                count_field_name = pipeline[1]["$count"]
                
                count = 0
                if "brand" in match_filter and "$regex" in match_filter["brand"]:
                    regex_val = match_filter["brand"]["$regex"].lower()
                    for doc in self.data_store.values():
                        if regex_val in doc.get("brand", "").lower():
                            count += 1
                    return [{count_field_name: count}]
                elif "category" in match_filter and "$regex" in match_filter["category"]:
                    regex_val = match_filter["category"]["$regex"].lower()
                    for doc in self.data_store.values():
                        if regex_val in doc.get("category", "").lower():
                            count += 1
                    return [{count_field_name: count}]

            elif pipeline and len(pipeline) > 0 and "$group" in pipeline[0]: # list categories
                 if pipeline[0]["$group"]["_id"] == "$category":
                    categories = {}
                    for doc in self.data_store.values():
                        cat = doc.get("category")
                        categories[cat] = categories.get(cat, 0) + 1
                    
                    output = []
                    # Check for projection
                    proj_stage = next((stage for stage in pipeline if "$project" in stage), None)
                    if proj_stage and proj_stage["$project"].get("category_name") == "$_id" and proj_stage["$project"].get("number_of_cheeses") == "$count":
                        for cat, num in categories.items():
                            output.append({"category_name": cat, "number_of_cheeses": num})
                        # Check for sort
                        sort_stage = next((stage for stage in pipeline if "$sort" in stage), None)
                        if sort_stage and "number_of_cheeses" in sort_stage["$sort"]:
                            output.sort(key=lambda x: x["number_of_cheeses"], reverse=sort_stage["$sort"]["number_of_cheeses"] == -1)
                        return output
                    else: # Default if no specific project stage for this mock
                        return [{"_id": cat, "count": num} for cat, num in categories.items()]

            print("(MockMongoCollection) Aggregation pipeline not fully mocked for this query. Returning empty.")
            return []
        
        def index_information(self): return {"text_search_index_text": True} 
        def create_index(self, keys, name=None, background=None, default_language=None): 
            print(f"(MockMongoCollection) Creating index {name} on {keys}")
            return name

    class MockMongoCursor: # Simplified cursor for MockMongoCollection.find
        def __init__(self, data, sort_criteria=None, limit_val=None):
            self.data = list(data) # Convert generator to list if it was one
            self.limit_val = limit_val
            if sort_criteria:
                # Simplified sort, assumes sort_criteria is like [('score', {'$meta': 'textScore'})]
                # This mock doesn't actually use the 'score' from $meta for sorting here.
                # keyword_search_mongo itself will sort by 'score' if it's a fallback text search.
                pass 

        def sort(self, sort_key_pairs):
            # This mock cursor's sort is a no-op as sorting is complex. 
            # The actual keyword_search_mongo handles sorting for fallback.
            # For LLM queries, sort should be part of the generated query if needed.
            return self
        
        def limit(self, num):
            self.limit_val = num
            # Apply limit immediately for __iter__ to work correctly
            if self.limit_val is not None:
                self.data = self.data[:self.limit_val]
            return self

        def __iter__(self):
            return iter(self.data)
        
        def __next__(self):
            return next(iter(self.data))
        
        def next(self):
            return self.__next__()

    # Instantiate mocks
    mock_embed_model = MockEmbeddingModel()
    mock_pinecone_idx = MockPineconeIndex()
    mock_mongo_coll = MockMongoCollection()

    # Test call
    test_query = "creamy french brie"
    print(f"\n--- Running Mock Test for query: '{test_query}' ---")

    # Updated MockLLM for keyword_search_mongo (which is now LLM-driven query generation)
    class MockMongoQueryGeneratingLLM:
        def invoke(self, messages):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            
            prompt_content = messages[0].content.lower() if messages and hasattr(messages[0], 'content') else ""

            if "mongo_aggregation_generation_prompt.txt" in prompt_content or "generate a mongodb aggregation pipeline" in prompt_content:
                print("(MockLLM for mongo_utils) Detected AGGREGATION prompt.")
                if "how many french cheeses" in prompt_content:
                    return MockResponse(content='''[{"\$match": {"brand": {"\$regex": "french", "\$options": "i"}}}, {"\$count": "total_french_cheeses"}]''')
                elif "list all categories" in prompt_content:
                    return MockResponse(content='''[{"\$group": {"_id": "\$category", "count": {"\$sum": 1}}}, {"\$project": {"category_name": "\$_id", "number_of_cheeses": "\$count", "_id": 0}}, {"\$sort": {"number_of_cheeses": -1}}]''')
                return MockResponse(content='''[]''') # Default empty pipeline
            
            elif "mongo_filter_generation_prompt.txt" in prompt_content or "generate a mongodb filter query document" in prompt_content:
                print("(MockLLM for mongo_utils) Detected FILTER prompt.")
                if "creamy french brie" in prompt_content:
                    return MockResponse(content='''{"\$and": [{"characteristics.texture": {"\$regex": "creamy", "\$options": "i"}}, {"brand": {"\$regex": "french", "\$options": "i"}}, {"name": {"\$regex": "brie", "\$options": "i"}}]}''')
                elif "goat cheese" in prompt_content:
                    return MockResponse(content='''{"characteristics.milk_type": {"\$regex": "goat", "\$options": "i"}}''')
                return MockResponse(content='''{}''') # Default empty filter
            
            print(f"(MockLLM for mongo_utils) Unhandled prompt for hybrid search test. Prompt: {prompt_content[:200]}")
            return MockResponse(content='''{}''') # Fallback for safety

    mock_llm_for_mongo = MockMongoQueryGeneratingLLM()

    # Test hybrid_cheese_search with the new mock LLM for MongoDB part
    results = hybrid_cheese_search(
        query=test_query,
        mongo_collection=mock_mongo_coll, 
        pinecone_index=mock_pinecone_idx,
        embedding_model=mock_embed_model,
        llm_instance=mock_llm_for_mongo, # This LLM is for keyword_search_mongo
        top_k_semantic=2,
        top_k_keyword=3 # keyword_search_mongo uses its own limit with LLM query
    )

    print(f"\n--- Mock Test Results for '{test_query}' ---")
    if results:
        for i, res in enumerate(results):
            print(f"{i+1}. Name: {res.get('name')}, SKU: {res.get('sku')}, "+
                  f"Semantic: {res.get('search_score_semantic', 'N/A')}, Keyword: {res.get('search_score_keyword', 'N/A')}")
    else:
        print("No results from hybrid search.")

    # Test an aggregation-style query for hybrid search
    agg_test_query = "how many french cheeses are there?"
    print(f"\n--- Running Mock Test for query: '{agg_test_query}' ---")
    results_agg = hybrid_cheese_search(
        query=agg_test_query,
        mongo_collection=mock_mongo_coll, 
        pinecone_index=mock_pinecone_idx, 
        embedding_model=mock_embed_model,
        llm_instance=mock_llm_for_mongo, 
        top_k_semantic=1, # Semantic might not be relevant for pure count
        top_k_keyword=1   # For aggregation, keyword_search_mongo handles its own logic
    )
    print(f"\n--- Mock Test Results for '{agg_test_query}' ---")
    if results_agg:
        for i, res in enumerate(results_agg):
            # If it was an aggregation like count, the structure would be different
            # keyword_search_mongo is designed to return a list of docs, or a list containing one summary doc.
            print(f"{i+1}. Result: {res}")
            if "total_french_cheeses" in res:
                print(f"   Count of French cheeses: {res['total_french_cheeses']}")
    else:
        print("No results from hybrid search for aggregation query.")

    # Test a query expected to primarily use filter + potentially fallback in mongo_utils mock
    fallback_test_query = "some rare cheese"
    # The mock LLM for mongo_utils might return {} for filter, triggering fallback if setup in mongo_utils
    # Let's make the mock LLM return an empty filter for this to test that path in hybrid_search
    # if mongo_utils.keyword_search_mongo falls back to text search.
    # The current `MockMongoQueryGeneratingLLM` will return {} for unknown filter prompts.
    print(f"\n--- Running Mock Test for query: '{fallback_test_query}' (expecting filter, maybe fallback) ---")
    results_fallback = hybrid_cheese_search(
        query=fallback_test_query,
        mongo_collection=mock_mongo_coll, 
        pinecone_index=mock_pinecone_idx, 
        embedding_model=mock_embed_model,
        llm_instance=mock_llm_for_mongo, 
        top_k_semantic=2,
        top_k_keyword=3
    )
    print(f"\n--- Mock Test Results for '{fallback_test_query}' ---")
    if results_fallback:
        for i, res in enumerate(results_fallback):
            print(f"{i+1}. Name: {res.get('name')}, SKU: {res.get('sku')}, "+
                  f"Semantic: {res.get('search_score_semantic', 'N/A')}, Keyword: {res.get('search_score_keyword', 'N/A')}")
    else:
        print("No results from hybrid search for fallback test query.")

    print("\nNote: The above test uses mocks. For real functionality, ensure Pinecone and MongoDB are connected and populated.") 