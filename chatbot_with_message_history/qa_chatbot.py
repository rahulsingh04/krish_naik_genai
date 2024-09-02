import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# LANGSMITH TRACKING
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Simple Q&A Chat-Bot With Ollama"

# Load The Groq API Key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name='gemma-2-9b-it')
print(llm)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    answer the questions based on the provided context only.
    Please provide the most accurate  response based on the question 
    <context>
    {context}
    <context>
    Question:{input}
    
    """
        )

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('pdf_directory')  # Data Ingestion Step
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
user_prompt = st.text_input("Enter Your Query From The Research Paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Data-Base Is Ready")

if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.time()  # Use time.time() for wall-clock time
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed_time = time.time() - start
        st.write(f"Response Time: {elapsed_time:.2f} seconds")

        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-------------------')
    else:
        st.write("Please load documents and create the vector embedding first.")
