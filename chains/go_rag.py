from infra.go_store import vertor_store
from infra.llm import llm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
human_message_prompt = HumanMessagePromptTemplate.from_template("You are a golang developer. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \nQuestion: {question} \nContext: {context} \nAnswer:")
rag_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
qa_chain = (
    {"context": vertor_store.get_retriever() | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        qa_chain.invoke(query)
