from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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
vector_store = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
retriever = vector_store.as_retriever()

def get_chunks(path):
    loader = DirectoryLoader(path, glob="*.pdf", show_progress=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len, 
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len (documents)} documents into {len (chunks)} chunks.")

    return chunks

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def create_rag_chain(llm, retriever):
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

def create_rag_chain_with_memory(llm, retriever):
    condense_q_system_prompt = """Given a chat history and the latest user question \
    which might reference the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    condense_q_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", condense_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        ) 
        | llm 
        | StrOutputParser()
    )

    qa_system_prompt = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    return (
        RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
        | qa_prompt
        | llm
    )

if __name__ == "__main__":
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    path = sys.argv[1]
    chunks = get_chunks(path)
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    db.persist()

    print(f"Stored {len (chunks)} chunks into {CHROMA_PATH}") 