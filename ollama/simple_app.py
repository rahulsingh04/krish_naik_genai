import os 
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

##LANGSMITH TRACKING
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistance. please response to the question asked"),
        ("user", "Question: {question}")
    ]
)




# input_text = "what is llm model "
# result = chain.invoke({"question":input_text})
# print(result)

### Streamlit Framework
st.title("Langchain Demo With gemma model")
input_text = st.text_area("What question do you have in your mind? ")

# importing gemma model
llm = Ollama(model = "gemma2:2b")
output_parser =  StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))