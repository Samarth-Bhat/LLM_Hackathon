from vectorDB_access import VectorDatabase  # Assuming the previous VectorDatabase class is saved as vector_db.py
import numpy as np

def calculate_similarity(term1, term2):
    # Initialize the vector database
    db = VectorDatabase(index_path="word_embeddings.index", metadata_path="word_metadata.json")
    
    try:
        # Get embeddings for both terms
        embedding1 = db.get_embedding(term1)
        embedding2 = db.get_embedding(term2)
        
        # Check if embeddings are found; if not, print an error message and return None
        if embedding1 is None or embedding2 is None:
            print(f"One or both terms not found in the database: {term1}, {term2}")
            return None
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return cosine_similarity
    
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None

# Example usage:
term1 = "amlodipine"
term2 = "olmesartan"
similarity = calculate_similarity(term1, term2)

if similarity is not None:
    print(f"Similarity between '{term1}' and '{term2}': {similarity}")
else:
    print("Could not calculate similarity due to errors.")
