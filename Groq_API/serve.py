from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()

from langserve import add_routes

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

from langchain_core.prompts import ChatPromptTemplate
generic_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")])

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

### Create Chain
chain = prompt|model|parser

# result = chain.invoke({"language":"hindi", "text":"where are you from "})
# print(result)

## App Definition
app = FastAPI(title="Langchain Serve", 
              version="1.0", 
              description="a simple api server using langchain runnable interfaces")

## Adding Chain Route

add_routes(
    app,
    chain,
    path= "/chain"
                    )

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)