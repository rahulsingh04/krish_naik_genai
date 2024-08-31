from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st 
import os 
from dotenv import load_dotenv
load_dotenv()

#### LANGSMITH TRACKING 

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "SIMPLE Q&A CHAT - BOT WITH OLLAMA"

#### Prompt Template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant . Please Response to the user query "),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, engine, temprature, max_tokens):
    llm = Ollama(model = engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer

### Title Of The App 
st.title("Enhance Q&A chat bot with open ai")

## sidebar for setting 
st.sidebar.title("Settings")

## select ollama model 

llm = st.sidebar.selectbox("Select ollama model",['gemma2:2b'])

## Adjust response  parameter 
temprature = st.sidebar.slider("Temprature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 50, max_value= 300, value= 150)

### Main interface for the user input 

st.write("GO Ahead And Ask Any Question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temprature, max_tokens)
    st.write(response)
else:
    st.write("Please Provide The Query")