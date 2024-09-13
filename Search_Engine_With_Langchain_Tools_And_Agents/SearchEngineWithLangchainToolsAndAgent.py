import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv

# ## Arxiv And wikipedia tools 
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

## duck-duck-go helps to search from the internet 
search = DuckDuckGoSearchResults(name = "Search")

st.title("Langchain - Chat With Search ")
'''
    In this example we are using StreamLit CallBack Handler to display the thought and actions of an
    agent in an interactive streamlit app. Try More Langchain  Streamlit Agent 
    example github.com/langchain-ai/streamlit-agent
    
    '''

## Side Bar for setting 
st.sidebar.title("Settings")

# Load The Groq API Key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

api_key = os.getenv('GROQ_API_KEY')

# api_key = st.sidebar.text_input("Enter The Groq Api Key ", type = 'password')


if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role":"assisstant", "content":"Hi, I am a Chatbot who can search on web . How can I help you ? "}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="What Is Machine Learning ?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key, model_name= "Gemma2-9b-It", streaming= True)
    tools = [search, arxiv, wiki]
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handling_parsing_errors = True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assisstant", "content":response})
        st.write(response)

