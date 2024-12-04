import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
headers = {"Authorization": "Bearer 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"}


def query_llm(augmented_query: str):
    """
    Sends the augmented query to the LLM API and gets the response.

    Args:
        augmented_query (str): The augmented query to be sent to the LLM.

    Returns:
        dict: The response from the LLM.
    """
    response = requests.post(API_URL, headers=headers, json={"inputs": augmented_query})
    return response.json()


def generate_new_query(prompt: str, similar_docs: list) -> str:
    """
    Combines the prompt with similar documents to create an augmented query.

    Args:
        prompt (str): The original prompt provided by the user.
        similar_docs (list): A list of similar documents retrieved from the database.

    Returns:
        str: The augmented query to be sent to the LLM.
    """
    context = "\n".join(similar_docs)
    return f"Context:\n{context}\n\nPrompt:\n{prompt}"
