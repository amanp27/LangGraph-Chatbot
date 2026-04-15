from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

prompt = PromptTemplate.from_template("{question}")

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({'question': "What is the capital of Mumbai?"})
print(result)