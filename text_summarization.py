import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get HuggingFace token from environment
HuggingFaceApi = os.getenv('HF_TOKEN')

# Function to generate the response
def generate_response(txt):
    # Ensure the HuggingFace API token is available
    if HuggingFaceApi is None:
        raise ValueError("HuggingFace API token is not set. Please set it in your environment variables.")

    # Instantiate the LLM model
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=150,
        temperature=0.7,
        token=HuggingFaceApi
    )

    # Split the text into chunks
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    
    # Create multiple documents from the split text
    docs = [Document(page_content=t) for t in texts]
    
    # Load the summarize chain for text summarization
    chain = load_summarize_chain(llm, chain_type='refine')
    
    # Run the summarization chain and return the result
    return chain.run(docs)

# Streamlit Page Setup
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Text input for the user
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submit_button = st.form_submit_button('Summarize')
    if submit_button:
        with st.spinner('Calculating...'):
            # Generate the response (summary)
            response = generate_response(txt_input)
            result.append(response)

# Display the result if available
if len(result):
    st.info(result[0])  # Display the first (and only) summary
