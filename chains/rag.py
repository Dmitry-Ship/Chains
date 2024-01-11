from infra.vector_store import retriever
from infra.llm import llm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = hub.pull("rlm/rag-prompt")
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        qa_chain.invoke(query)
