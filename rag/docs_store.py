from langchain_community.embeddings import HuggingFaceEmbeddings
from .vectore import VectorStore, get_chunks_from_docs
import sys

embeddings = HuggingFaceEmbeddings()
blindsight_store = VectorStore(embeddings, "chroma/blindsight")

if __name__ == "__main__":
    path = sys.argv[1]
    chunks = get_chunks_from_docs(path)
    blindsight_store.store(chunks) 
