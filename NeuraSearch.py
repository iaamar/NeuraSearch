from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
import tiktoken
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from groq import Groq
import pinecone
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize embedding model, Pinecone and Groq
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
index_name = "ragvectorize-index"
namespace = "sample-doc"

# st.title("Document Similarity with Pinecone and Langchain")
# st.write("This app allows you to process PDFs, calculate sentence similarity, and use Pinecone for document embeddings.")

# # Text input from the user
# text_input = st.text_area("Enter text for embedding:")

# if text_input:
#     query_result = embeddings.embed_query(text_input)
#     st.write(f"Embedding result length: {len(query_result)}")

# # Cosine similarity calculation
# st.subheader("Cosine Similarity Between Sentences")
# sentence1 = st.text_input("Enter Sentence 1", "I like walking to the park")
# sentence2 = st.text_input("Enter Sentence 2", "I like running to the office")

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# def cosine_similarity_between_sentences(sentence1, sentence2):
#     embedding1 = np.array(get_huggingface_embeddings(sentence1))
#     embedding2 = np.array(get_huggingface_embeddings(sentence2))

#     embedding1 = embedding1.reshape(1, -1)
#     embedding2 = embedding2.reshape(1, -1)

#     similarity = cosine_similarity(embedding1, embedding2)
#     return similarity[0][0]

# if st.button("Calculate Similarity"):
#     similarity = cosine_similarity_between_sentences(sentence1, sentence2)
#     st.write(f"Cosine similarity between '{sentence1}' and '{sentence2}': {similarity:.4f}")

# Streamlit UI
st.title("Neura Search")
# st.write("Upload a PDF file to vectorize and store in Pinecone")
# Upload PDF
uploaded_file = st.file_uploader("", type="pdf")
document_data = []

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("/tmp", uploaded_file.name)  # Save to a temp directory
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())    
    loader = PyPDFLoader(temp_file_path)
    document_data = loader.load()

# Prepare the text for embedding and use a separate list for storing Documents
document_objects = []

for document in document_data:
    # Access metadata and page_content directly from the Document object
    document_source = document.metadata.get('source', 'Unknown source')
    document_content = document.page_content if document.page_content else "No content available"     
    # Create a Document object with the required format
    doc = Document(page_content=f"<Source>\n{document_source}\n</Source>\n\n<Content>\n{document_content}\n</Content>")
    document_objects.append(doc)

# Create a PineconeVectorStore from the document objects
vectorstore_from_documents = PineconeVectorStore.from_documents(
    document_objects,
    embeddings,
    index_name=index_name,
    namespace=namespace
)

# Initialize the Pinecone index
pincone_index = pc.Index(index_name)

query = st.text_input("Enter a query to search:")

if query and st.button("Search"):

    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pincone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=10,
        include_metadata=True,
        namespace=namespace
    )

    # Extract contexts from the matches and handle cases where metadata is missing
    contexts = []
    for items in top_matches["matches"]:
        if "metadata" in items:
            context_text = items["metadata"].get("content", "No content available")
            contexts.append(context_text)
    
    if contexts:
        augmented_query = "«CONTEXT>|n" + "\n\n-------|n|n".join(contexts[:10]) + "\n-------|n</CONTEXT>\n\nMY QUESTION: " + query

        # Perform the augmented generation with Groq API
        system_prompt = """You are an expert at understanding and analyzing company data - particularly shipping orders, purchase orders, invoices, and inventory reports. Answer any questions I have, based on the data provided. Always consider all of the context provided when forming a response."""

        # Make the API call to Groq for text completion
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ],
        )
        response = response.choices[0].message.content

        # Display the generated response
        st.write(response)