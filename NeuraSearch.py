# streamlit_app.py
import os
import streamlit as st
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Set up API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize Pinecone
# Create a Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, if not create it
index_name = "ragpipe"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Make sure this matches the embedding dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'  # Choose the region for your Pinecone setup
        )
    )

# Sidebar for input
st.sidebar.title("API Key Configuration")
st.sidebar.text_input("OpenAI API Key", type="password", value=openai_api_key)
st.sidebar.text_input("Pinecone API Key", type="password", value=pinecone_api_key)

# Initialize HuggingFace Embeddings client
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("Document Similarity with Pinecone and Langchain")
st.write("This app allows you to process PDFs, calculate sentence similarity, and use Pinecone for document embeddings.")

# Text input from the user
text_input = st.text_area("Enter text for embedding:")

if text_input:
    query_result = embeddings.embed_query(text_input)
    st.write(f"Embedding result length: {len(query_result)}")

# Cosine similarity calculation
st.subheader("Cosine Similarity Between Sentences")
sentence1 = st.text_input("Enter Sentence 1", "I like walking to the park")
sentence2 = st.text_input("Enter Sentence 2", "I like running to the office")

if st.button("Calculate Similarity"):
    def get_huggingface_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)
        return model.encode(text)

    def cosine_similarity_between_sentences(sentence1, sentence2):
        embedding1 = np.array(get_huggingface_embeddings(sentence1))
        embedding2 = np.array(get_huggingface_embeddings(sentence2))

        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)

        similarity = cosine_similarity(embedding1, embedding2)
        return similarity[0][0]

    similarity = cosine_similarity_between_sentences(sentence1, sentence2)
    st.write(f"Cosine similarity between '{sentence1}' and '{sentence2}': {similarity:.4f}")

# Directory processing
st.subheader("Process PDF Directory")
directory_path = st.text_input("Enter directory path for PDF files")

def process_directory(directory_path):
    data = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            st.write(f"Processing file: {file_path}")
            loader = PyPDFLoader(file_path)
            data.append({"File": file_path, "Data": loader.load()})
    return data

if directory_path:
    documents = process_directory(directory_path)

    st.write("Documents loaded:")
    for document in documents:
        st.write(document['File'])

# Pinecone Index initialization
st.subheader("Pinecone Setup")
index_name = st.text_input("Enter Pinecone Index Name", "ragpipe")
namespace = st.text_input("Enter Namespace", "company-documents")

if st.button("Initialize Pinecone Index"):
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    st.write(f"Pinecone Index '{index_name}' initialized.")

# Query Pinecone
query = st.text_input("Enter query for Pinecone")

if query and st.button("Query Pinecone"):
    raw_query_embedding = get_huggingface_embeddings(query)
    pincone_index = pc.Index(index_name)

    top_matches = pincone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=10,
        include_metadata=True,
        namespace=namespace
    )

    contexts = [items["metadata"]["text"] for items in top_matches["matches"]]
    st.write(contexts)

# Display final augmented query and response
st.subheader("Augmented Query & LLM Response")
augmented_query = f"CONTEXT:\n\n{'\n-------\n'.join(contexts[:10])}\n\nMY QUESTION: {query}"
st.write(augmented_query)

# Mock response for now (replace with actual API call to Groq)
st.write("LLM response goes here")
