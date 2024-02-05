from langchain_community.vectorstores import Chroma
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