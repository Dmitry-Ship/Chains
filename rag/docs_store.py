from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from .vectore import VectorStore
from dotenv import load_dotenv
import sys

load_dotenv()
embeddings = HuggingFaceEmbeddings()

def get_chunks_from_docs(path):
    loader = DirectoryLoader(path, glob="*.pdf", show_progress=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len, 
    )

    return  text_splitter.split_documents(documents)

vector_store = VectorStore(embeddings, "chroma")

if __name__ == "__main__":
    path = sys.argv[1]
    chunks = get_chunks_from_docs(path)
    vector_store.store(chunks) 