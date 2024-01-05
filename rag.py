from utils.rag import create_rag_chain, retriever
from utils.llm import llm

rag_chain = create_rag_chain(llm, retriever)

while True:
    query = input("\nUser: ")
    rag_chain.invoke({
        "question": query, 
    })
