from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from .vectore import VectorStore
from dotenv import load_dotenv
import sys

load_dotenv()
embeddings = HuggingFaceEmbeddings()

def get_chunks_from_code(path):
    loader = GenericLoader.from_filesystem(
        path,
        glob="**/*",
        exclude=[
            '*_test.go', 
            '*_gen.go', 
            '*_generated.go',
            '*layer.go', 
            '*_tests/*', 
            '*/opentracing/*', 
            '*/channels/store/*', 
            "mocks/*",
        ],
        suffixes=[".go"],
        parser=LanguageParser(),
    )
    documents = loader.load()

    go_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.GO, chunk_size=1000, chunk_overlap=200
    )

    return  go_splitter.split_documents(documents)

vertor_store = VectorStore(embeddings, "chroma_code")

if __name__ == "__main__":
    path = sys.argv[1]
    chunks = get_chunks_from_code(path)
    vertor_store.store(chunks) 