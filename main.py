import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import AgentExecutor, Tool

# Initialize the LLM
llm = OpenAI(model="text-davinci-003", temperature=0.5)

# Load Dataset
def load_data():
    loader = JSONLoader('path_to_your_json_directory', glob="*.json")
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = load_data()

# RAG Setup
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever)

# Streamlit GUI
st.title("Pharma Knowledge Assistant")
st.sidebar.header("Features")
features = st.sidebar.radio(
    "Choose a Feature:",
    ("Question Answering", "Recommender", "Summarizer", "Alternatives Generator")
)

if features == "Question Answering":
    st.header("Ask about Pharmaceutical Products")
    user_query = st.text_input("Enter your question:")
    if user_query:
        response = qa_chain.run(user_query)
        st.write(f"**Response:** {response}")

elif features == "Recommender":
    st.header("Get Medication Recommendations")
    user_symptoms = st.text_input("Enter your symptoms or conditions:")
    if user_symptoms:
        response = qa_chain.run(f"Recommend medication for: {user_symptoms}")
        st.write(f"**Recommendation:** {response}")

elif features == "Summarizer":
    st.header("Summarize Product Details")
    product_name = st.text_input("Enter the product name:")
    if product_name:
        response = qa_chain.run(f"Summarize details for {product_name}")
        st.write(f"**Summary:** {response}")

elif features == "Alternatives Generator":
    st.header("Find Alternatives")
    medication = st.text_input("Enter the medication name:")
    if medication:
        response = qa_chain.run(f"Suggest alternatives for {medication}")
        st.write(f"**Alternatives:** {response}")

# Metrics and Evaluation
st.sidebar.subheader("Metrics")
if st.sidebar.button("Evaluate Model"):
    st.sidebar.write("Metrics evaluation coming soon.")

# Instructions
st.sidebar.markdown(
    """
    **Instructions:**  
    - Use the sidebar to select a feature.  
    - Enter the required input in the main panel.  
    - The assistant will provide relevant outputs.  
    """
)
