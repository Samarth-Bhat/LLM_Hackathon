# embedding_utils.py
import json
import string
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import faiss
import nltk

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

def extract_values(data):
    """Recursively extract all text values from a JSON object."""
    values = []
    if isinstance(data, dict):
        for value in data.values():
            values.extend(extract_values(value))
    elif isinstance(data, list):
        for item in data:
            values.extend(extract_values(item))
    elif isinstance(data, (str, int, float)):
        values.append(str(data))  # Convert all values to strings
    return values

def preprocess_text(text):
    """Clean and tokenize text."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return tokens

def process_input_data(input_data):
    """Process input data (JSON or plain text) and extract sentences."""
    all_sentences = []
    
    if isinstance(input_data, str):  # If input_data is a path to a file
        # Check if it's a JSON file or a plain text file
        if input_data.endswith(".json"):
            try:
                with open(input_data, "r") as f:
                    json_data = json.load(f)
                    # Extract all text values and preprocess them
                    values = extract_values(json_data)
                    for value in values:
                        tokens = preprocess_text(value)
                        if tokens:
                            all_sentences.append(tokens)
            except json.JSONDecodeError as e:
                print(f"Error decoding {input_data}: {e}")
        else:  # Assuming it's a plain text file
            with open(input_data, "r") as f:
                text = f.read()
                tokens = preprocess_text(text)
                if tokens:
                    all_sentences.append(tokens)
    elif isinstance(input_data, list):  # If input_data is a list of sentences
        for sentence in input_data:
            tokens = preprocess_text(sentence)
            if tokens:
                all_sentences.append(tokens)
    else:
        print("Unsupported input type. Please provide a path to a JSON or text file, or a list of sentences.")
    
    return all_sentences

def train_word2vec(sentences):
    """Train a Word2Vec model on the sentences."""
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    print("Word2Vec model trained.")
    return model

def create_faiss_index(word_vectors):
    """Create a FAISS index for the word vectors."""
    print("Extracting word vectors and preparing FAISS index...")
    embeddings = np.array([word_vectors[word] for word in word_vectors.index_to_key])
    word_list = word_vectors.index_to_key  # List of words corresponding to vectors

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    print(f"Initializing FAISS index with {dimension}-dimensional vectors...")
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric
    index.add(embeddings)  # Add vectors to the index
    print(f"Added {embeddings.shape[0]} embeddings to the FAISS index.")
    
    return index, word_list
