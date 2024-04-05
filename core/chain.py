from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt = PromptTemplate(
    template="{input}",
    input_variables=['input']
)

question_chain = LLMChain(
    llm=llm,
    prompt=prompt
)
