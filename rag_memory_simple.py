from utils.rag import retriever
from utils.llm import llm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

rag_chain_with_memory = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        rag_chain_with_memory.invoke({
            "question": query, 
        })
