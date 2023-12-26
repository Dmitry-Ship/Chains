from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from operator import itemgetter
from dotenv import load_dotenv
import shutil
import os
import sys

load_dotenv()

CHROMA_PATH = "chroma"
embeddings = HuggingFaceEmbeddings()

def save_documents_to_db(path):
    loader = DirectoryLoader(path, glob="*.pdf", show_progress=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len, 
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len (documents)} documents into {len (chunks)} chunks.")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    db.persist()

    print(f"Stored {len (chunks)} chunks into {CHROMA_PATH}") 



def create_rag_chain(llm):
    vector_store = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
    retriever = vector_store.as_retriever()
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", template),
    ])
    return (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": RunnablePassthrough(), 
        "system_message": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    path = sys.argv[1]
    save_documents_to_db(path)