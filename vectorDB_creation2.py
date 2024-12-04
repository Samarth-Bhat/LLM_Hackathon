# vectorDB_access.py
import json
import os
import faiss
import numpy as np
from embedding_utils import process_input_data, train_word2vec, create_faiss_index

def load_json_data(file_path):
    """Load JSON data and extract text for embedding."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process the JSON data to extract and tokenize text
    sentences = process_input_data(data)
    return sentences

def create_embeddings_and_index(sentences):
    """Generate word embeddings using Word2Vec and create FAISS index."""
    # Train Word2Vec model
    model = train_word2vec(sentences)
    
    # Create FAISS index
    index, word_list = create_faiss_index(model.wv)
    return model, index, word_list

def process_all_json_files_in_directory(directory_path):
    """Process all JSON files in the given directory and add them to the vector database."""
    all_sentences = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        print(f"Processing file: {file_path}")
        
        # Load JSON data and process it
        sentences = load_json_data(file_path)
        all_sentences.extend(sentences)
    
    # Generate embeddings and create index from all processed sentences
    model, index, word_list = create_embeddings_and_index(all_sentences)
    print("Data from all JSON files added to vector database (FAISS index).")
    
    return model, index, word_list

def save_faiss_index(index, index_file_path):
    """Save the FAISS index to a file."""
    faiss.write_index(index, index_file_path)
    print(f"FAISS index saved to {index_file_path}")

def save_metadata(word_list, metadata_file_path):
    """Save word metadata to a JSON file."""
    metadata = {
        "words": word_list,
        "total_words": len(word_list)
    }
    
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_file_path}")

def add_to_vector_db(directory_path):
    """Process all JSON files in the directory and store embeddings in FAISS."""
    # Process all JSON files in the directory and create embeddings/index
    model, index, word_list = process_all_json_files_in_directory(directory_path)
    
    # Save FAISS index
    save_faiss_index(index, 'word_embeddings.index')
    
    # Save word metadata
    save_metadata(word_list, 'word_metadata.json')
    
    print("Vector database and metadata have been saved.")
    return index, word_list

# Example usage
directory_path = 'datasets/microlabs_usa'

# Process all JSON files in the directory and add them to the vector database
index, word_list = add_to_vector_db(directory_path)
