from infra.llm import llm
from .docs_store import vector_store
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain import hub

rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
rephrase_chain = create_history_aware_retriever(llm, vector_store.get_retriever(), rephrase_prompt)

prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("""
    Answer the question based on the below context.
    Context: {context}
    Question: {input}
    Answer:""")
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(rephrase_chain, document_chain)
memory = ConversationBufferMemory()

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        ai_response = retrieval_chain.invoke({
            "chat_history": memory.load_memory_variables({})['history'],
            "input": query
        })

        memory.save_context({"input": query}, {"output": ai_response['answer']})