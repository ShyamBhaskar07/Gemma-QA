import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Demo")

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    PLease provide the most accurate response based on question
    <context>
    {context}
    Questions : {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize the embedding model
        st.session_state.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("../data/")
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.error("No documents found in the directory. Please check the path.")
            return

        # Split the documents into smaller chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        if not st.session_state.final_documents:
            st.error("Document splitting failed. Please check the documents.")
            return

        # Extract the text content from each Document object
        documents_text = [doc.page_content for doc in st.session_state.final_documents]

        # Generate embeddings for the document chunks
        embeddings = st.session_state.embeddings_model.embed_documents(documents_text)

        if not embeddings:
            st.error("Failed to generate embeddings. Please check the embedding model.")
            return

        # Create the vector store from the embeddings and documents
        st.session_state.vectors = FAISS.from_texts(documents_text, st.session_state.embeddings_model)


prompt1 = st.text_input("Enter your question from documents")

if st.button("Creating vector store"):
    vector_embedding()
    st.write("Vector store created")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retreival_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retreival_chain.invoke({'input':prompt1})
    st.write(response['answer'])


