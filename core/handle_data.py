from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from pathlib import Path
import shutil


class DataHandler:
    """Class responsible for loading data, splitting it into chunks, transforming it
     into embeddings and saving into vectorstore"""

    def __init__(
            self,
            text_splitter: TextSplitter,
            loader: type[UnstructuredBaseLoader],
            embeddings: Embeddings
    ):
        self.text_splitter = text_splitter
        self.loader = loader
        self.embeddings = embeddings
        self.vectorstore_path: str = (Path.cwd() / "vectorstore").__str__()

    def save_docs_to_vectorstore(self, files: list[Path]) -> bool:
        if not files:
            return False
        shutil.rmtree(self.vectorstore_path, ignore_errors=True)
        for file in files:
            loader = self.loader(file.__str__())
            docs = loader.load_and_split(
                text_splitter=self.text_splitter
            )
            _ = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.vectorstore_path
            )
        return True

    def load_docs_from_vectorstore(self) -> VectorStore:
        db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vectorstore_path
            )
        return db
