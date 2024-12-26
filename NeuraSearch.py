from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Streamlit app
st.title("Neura Search")

# Retrieve API keys from secrets or environment variables
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
groq_api_key = st.secrets.get["GROQ"]

index_name = "ragvectorize-index"
namespace = "sample-doc"

if not pinecone_api_key:
    st.error("Pinecone API Key is missing. Please configure it in your environment or Streamlit secrets.")
    st.stop()

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)
pinecone_index = pinecone.Index(index_name)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file for processing:", type="pdf")
document_objects = []

if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process the PDF
    loader = PyPDFLoader(temp_file_path)
    document_data = loader.load()

    for document in document_data:
        # Access metadata and page content
        document_source = document.metadata.get("source", "Unknown source")
        document_content = document.page_content or "No content available"
        
        # Create Document object
        doc = Document(
            page_content=f"<Source>\n{document_source}\n</Source>\n\n<Content>\n{document_content}\n</Content>"
        )
        document_objects.append(doc)

    # Create Pinecone vector store from document objects
    PineconeVectorStore.from_documents(
        document_objects,
        embeddings,
        index_name=index_name,
        namespace=namespace
    )
    st.success("Document processed and indexed successfully!")

# Search query input
query = st.text_input("Enter a query to search:")
if query and st.button("Search"):
    # Get embedding for the query
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query)

    # Perform search in Pinecone
    results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=10,
        include_metadata=True,
        namespace=namespace
    )

    # Extract and display results
    contexts = [
        item["metadata"].get("content", "")
        for item in results.get("matches", [])
    ]

    if contexts:
        st.write("Search Results:")
        for i, context in enumerate(contexts, start=1):
            # st.write(f"**Result {i}:**")
            st.write(context)

        # Augment query with context for Groq API (if available)
        if groq_api_key:
            from groq import Groq

            groq_client = Groq(api_key=groq_api_key)
            augmented_query = (
                "«CONTEXT>|n"
                + "\n\n-------|n|n".join(contexts[:10])
                + "\n-------|n</CONTEXT>\n\nMY QUESTION: "
                + query
            )

            system_prompt = """You are an expert at understanding and analyzing company data - particularly shipping orders, purchase orders, invoices, and inventory reports. Answer any questions I have, based on the data provided. Always consider all of the context provided when forming a response."""

            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": augmented_query},
                ],
            )
            # st.write("Groq Response:")
            st.write(response.choices[0].message.content)
    else:
        st.warning("No matching results found.")
