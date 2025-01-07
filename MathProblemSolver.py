import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.callbacks import StreamlitCallbackHandler

## Set Up the Streamlit App
st.set_page_config(page_title='Text To Math Problem Solver And Data Search Assistant ')
st.title('Text To Math Problem Solver Using Google Gemma 2')

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model='Gemma2-9b-It', groq_api_key = groq_api_key)

### Initializing The Tool

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = 'wikipedia',
    func=wikipedia_wrapper.run,
    description = 'A tool for searching the Internet to find the various information on the topic mentioned'
)

### Initialize  Math Tools

math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = 'calculator',
    func = math_chain.run,
    description= 'A tool for answering math related questions. Only Input Mathematical expression needs to be provided'
)

prompt = ''' You are agent tasked for solving users mathematical question . logically arrived at the sollution and provide 
            a detailed explanation at point wise .
            And display it point wise for the question below 
            Question : {question}
            Answer   :  
            '''

prompt_template = PromptTemplate(
    input_variables= ['question'], 
    template = prompt
)

### Combine All The Tools Into Chains 

chain = LLMChain(llm = llm , prompt = prompt_template)

reasoning_tool = Tool(
    name = 'Reasoning tool',
    func = chain.run,
    description= 'A tool for answering logic based and answer based questions '
)

## Initialized The Agents
assistant_agent = initialize_agent(
    tools= [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {'role':'assistant', 'content': "Hi i am a math chatbot who can answer all your question related to the math"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

## Function to generate response 
def generate_response(question):
    response = assistant_agent.invoke({'input':question})
    return response

### Let's start the interaction 
question = st.text_area("Enter Your Question ", "I have 2 apple 3 banana what is the sum of this ?")
if st.button('find my answer'):
    if question:
        with st.spinner("Generate response... "):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant', 'content':response})
            st.write("### Response : ")
            st.success(response)

    else:
        st.warning("Please Write The Question ")