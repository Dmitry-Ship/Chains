from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import shutil
import os

class VectorStore:
    def __init__(self, embeddings, store_path,):
        self.store_path = store_path
        self.embeddings = embeddings
    
    def store(self, chunks):
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
        db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.store_path
        )

        print(f"Stored {len (chunks)} chunks into {self.store_path}")
    
    def get_retriever(self):
        vector_store = Chroma(embedding_function=self.embeddings, persist_directory=self.store_path)
        retriever = vector_store.as_retriever()
        return retriever

def get_chunks_from_docs(path):
    loader = DirectoryLoader(path, glob="*.pdf", show_progress=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len, 
    )

    return text_splitter.split_documents(documents)
