import streamlit as st
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from groq import Groq
import pinecone
import numpy as np
import os

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
index_name = "ragvectorize-index"
namespace = "sample-doc"

# Initialize HuggingFace Embeddings
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

# Streamlit UI
st.title("PDF Vectorization, Retrieval, and Augmented Generation with RAG")
st.write("Upload a PDF file to vectorize and store in Pinecone")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("/tmp", uploaded_file.name)  # Save to a temp directory
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully!")
    
    # Process the PDF
    st.write("Processing the PDF and vectorizing it...")
    loader = PyPDFLoader(temp_file_path)
    document_data = loader.load()

    # Prepare the text for embedding
    for doc in document_data:
        document_source = doc.metadata.get('source', 'unknown')
        document_content = doc.page_content
        st.write(f"Processing Document: {document_source}")
        st.write(document_content[:500])  # Show part of the document content

        # Get embeddings
        embedding = embeddings.embed_query(document_content)
        
        # Prepare the document data for Pinecone
        doc_data = {"content": document_content, "embedding": embedding}
        
        # Connect to Pinecone and store the document
        pinecone_index = pc.Index(index_name)
        pinecone_index.upsert([(f"{document_source}", embedding)], namespace=namespace)

    st.success("Document has been vectorized and stored in Pinecone!")

    # Query Section
    query = st.text_input("Enter a query to search and generate an augmented response:")

    if query:
        # Vectorize the query
        query_embedding = embeddings.embed_query(query)
        
        # Query Pinecone
        query_results = pinecone_index.query(query_embedding, top_k=5, namespace=namespace)

        st.write("Top Matches:")
        contexts = []
        for match in query_results['matches']:
            context_text = match['metadata']['content']
            contexts.append(context_text)
            st.write(f"Match ID: {match['id']}")
            st.write(f"Similarity Score: {match['score']}")
            st.write(f"Content: {context_text[:500]}...")

        # Construct the augmented query
        augmented_query = "CONTEXT:\n" + "\n\n".join(contexts) + f"\n\nQUESTION: {query}"

        st.write("Generated Augmented Query:")
        st.write(augmented_query)

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
        st.write("Generated Response:")
        st.write(response)
