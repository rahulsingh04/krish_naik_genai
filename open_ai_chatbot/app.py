import streamlit as st 
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  ChatPromptTemplate



import os 
from dotenv import load_dotenv
load_dotenv()

### LANGSMITH TRACKING 
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "QA CHAT BOT WITH OPEN AI "

### PROMPT TEMPLATE 

prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temprature , max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model = llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer

## Title Of The App 

st.title("Enhance question chatbot with openai")

## sidebar for setting 
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Your Open AI API Key ", type = 'password')

## Drop down and select various llm model 
llm = st.sidebar.selectbox("Select An Opne AI Model", ["gpt-4o", 'gpt-4-turbo', 'gpt-4'])


## Adjust response  parameter 
temprature = st.sidebar.slider("Temprature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 50, max_value= 300, value= 150)


### Main interface for the user input 

st.write("GO Ahead And Ask Any Question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temprature, max_tokens)
    st.write(response)
else:
    st.write("Please Provide The Query")