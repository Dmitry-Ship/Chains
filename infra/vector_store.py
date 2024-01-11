from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import shutil
import os
import sys

load_dotenv()

CHROMA_PATH = "chroma"
embeddings = HuggingFaceEmbeddings()
vector_store = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
retriever = vector_store.as_retriever()

def get_chunks_from_docs(path):
    loader = DirectoryLoader(path, glob="*.pdf", show_progress=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len, 
    )

    return  text_splitter.split_documents(documents)

if __name__ == "__main__":
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    path = sys.argv[1]
    chunks = get_chunks_from_docs(path)
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    db.persist()

    print(f"Stored {len (chunks)} chunks into {CHROMA_PATH}") 