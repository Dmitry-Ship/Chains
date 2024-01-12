# from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)

repo_path = "/Users/dmitryshipunov/GitLab/backend"

loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".go"],
    parser=LanguageParser(),
)
documents = loader.load()

go_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.GO, 
    chunk_size=1000, 
    chunk_overlap=200,
)

chunks = go_splitter.split_documents(documents)

print("\n\n--8<--\n\n".join([document.page_content for document in chunks]))