import os
import json
import re
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import nltk
import faiss
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK resources: {e}")


class JSONTextProcessor:
    def __init__(
            self,
            chunk_size: int = 500,
            embedding_model: str = 'all-MiniLM-L6-v2',
            max_tokens: int = 512
    ):
        """
        Initialize the JSON Text Processor.

        Args:
            chunk_size (int): Maximum number of characters per chunk
            embedding_model (str): SentenceTransformer model to use
            max_tokens (int): Maximum tokens for embedding
        """
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens

        # Explicitly set device to CPU
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device='cpu'
        )

        # Set stop words
        self.stop_words = set(stopwords.words('english'))

    def load_json_files(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load all JSON files from a given directory.

        Args:
            directory (str): Path to the directory containing JSON files

        Returns:
            List[Dict[str, Any]]: List of loaded JSON data
        """
        json_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        json_files.append(json.load(file))
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.error(f"Error processing {filename}: {e}")

        return json_files

    def extract_text_recursively(
            self,
            data: Any,
            depth: int = 0,
            max_depth: int = 5
    ) -> List[str]:
        """
        Recursively extract text from nested structures with depth control.

        Args:
            data (Any): Input data to extract text from
            depth (int): Current recursion depth
            max_depth (int): Maximum recursion depth

        Returns:
            List[str]: Extracted text chunks
        """
        if depth > max_depth:
            return []

        extracted_texts = []

        def _extract(item):
            if isinstance(item, dict):
                for value in item.values():
                    extracted_texts.extend(
                        self.extract_text_recursively(
                            value, depth + 1, max_depth
                        )
                    )
            elif isinstance(item, list):
                for elem in item:
                    extracted_texts.extend(
                        self.extract_text_recursively(
                            elem, depth + 1, max_depth
                        )
                    )
            elif isinstance(item, (str, int, float)):
                text = str(item).strip()
                if text and len(text) > 3:
                    # Break long texts into chunks
                    for chunk in self._chunk_text(text):
                        extracted_texts.append(chunk)

        _extract(data)
        return extracted_texts

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split long text into manageable chunks.

        Args:
            text (str): Input text to chunk

        Returns:
            List[str]: Text chunks
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords(self, text: str) -> str:
        """
        Remove stop words from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with stop words removed
        """
        words = word_tokenize(text)
        filtered_words = [
            word for word in words
            if word.lower() not in self.stop_words
        ]
        return ' '.join(filtered_words)

    def process_texts(self, texts: List[str]) -> List[str]:
        """
        Process texts: clean and remove stop words.

        Args:
            texts (List[str]): Input texts

        Returns:
            List[str]: Processed texts
        """
        processed_texts = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            processed_text = self.remove_stopwords(cleaned_text)
            if processed_text:
                processed_texts.append(processed_text)

        return processed_texts

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for texts.

        Args:
            texts (List[str]): Input texts

        Returns:
            np.ndarray: Embeddings
        """
        # Truncate texts to max tokens
        truncated_texts = [
            ' '.join(text.split()[:self.max_tokens])
            for text in texts
        ]

        return self.embedding_model.encode(
            truncated_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def create_faiss_index(
            self,
            embeddings: np.ndarray
    ) -> Optional[faiss.IndexFlatL2]:
        """
        Create a FAISS index for embeddings.

        Args:
            embeddings (np.ndarray): Input embeddings

        Returns:
            faiss.IndexFlatL2 or None
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings to index")
            return None

        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)

            # Normalize vectors for better cosine-like similarity
            faiss.normalize_L2(embeddings)

            index.add(embeddings)
            return index
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return None

    def process_directory(
            self,
            directory: str,
            output_index_path: str = 'text_index.faiss',
            output_texts_path: str = 'processed_texts.json'
    ):
        """
        Process all JSON files in a directory.

        Args:
            directory (str): Input directory with JSON files
            output_index_path (str): Path to save FAISS index
            output_texts_path (str): Path to save processed texts
        """
        # Load JSON files
        json_files = self.load_json_files(directory)

        # Extract texts from all files
        all_raw_texts = []
        for json_data in json_files:
            all_raw_texts.extend(
                self.extract_text_recursively(json_data)
            )

        logger.info(f"Extracted {len(all_raw_texts)} raw text chunks")

        # Process texts
        processed_texts = self.process_texts(all_raw_texts)

        logger.info(f"Processed {len(processed_texts)} texts")

        # Create embeddings
        embeddings = self.create_embeddings(processed_texts)

        # Create FAISS index
        faiss_index = self.create_faiss_index(embeddings)

        if faiss_index:
            # Save index and texts
            faiss.write_index(faiss_index, output_index_path)
            with open(output_texts_path, 'w', encoding='utf-8') as f:
                json.dump(processed_texts, f, ensure_ascii=False)

            logger.info(f"Saved FAISS index to {output_index_path}")
            logger.info(f"Saved processed texts to {output_texts_path}")
        else:
            logger.error("Failed to create FAISS index")


def main():
    # Example usage
    processor = JSONTextProcessor(
        chunk_size=500,  # Adjust based on your text size
        embedding_model='all-MiniLM-L6-v2',
        max_tokens=512
    )

    # Replace with your directory path
    input_directory = '/datasets/microlabs_usa'
    processor.process_directory(input_directory)


if __name__ == "__main__":
    main()