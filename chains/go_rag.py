from infra.llm import llm
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from .rag import RagMemoryChain
from infra.go_store import vertor_store

human_message_prompt = HumanMessagePromptTemplate.from_template("""
You are a golang developer. Use the following pieces of retrieved codebase snippets to answer the question.
If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:
""")
rag_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
memory = ConversationSummaryBufferMemory(llm=llm)


rag_memory_chain = RagMemoryChain(llm=llm, memory=memory, rag_prompt=rag_prompt, retriever=vertor_store.get_retriever())

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        rag_memory_chain.invoke(query)
