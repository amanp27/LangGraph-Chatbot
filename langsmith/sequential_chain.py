from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Sequential Chain Example"


llm1 = ChatOpenAI(model = "gpt-4o-mini", temperature=0.7)
llm2 = ChatOpenAI(model = "gpt-4o", temperature=0.3)

prompt1 = PromptTemplate(
    template= "Generate a detail report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate a five pointer summary from the following text \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | llm1 | parser | prompt2 | llm2 | parser

config = {'run_name': 'sequentail_chain',
          'tags': ['sequential_chain'],
          'metadata': {'example': 'This is example of sequential chain'}}


result = chain.invoke({'topic': "Indian IT Industry"})

print(result)