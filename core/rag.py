from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from core.chain import Chain
from core.handle_data import DataHandler

chain = Chain(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    prompt=PromptTemplate(
        template="""Use the following context to answer the question at the end.
            {context}.
            {question}""",
        input_variables=['context', 'question']
    )
)

data_handler = DataHandler(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ),
    loader=UnstructuredPDFLoader,
    embeddings=OpenAIEmbeddings()
)
