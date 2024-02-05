from infra.llm import llm
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from infra.go_store import vertor_store
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
rephrase_chain = create_history_aware_retriever(llm, vertor_store.get_retriever(), rephrase_prompt)

prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("""
    You are a golang developer. Use codebase snippets to answer the question.
    If you don't know the answer, just say that you don't know.
    Snippets: {context}
    Question: {question}
    Answer:""")
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(rephrase_chain, document_chain)

memory = ConversationSummaryBufferMemory(llm=llm)
if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        ai_response = retrieval_chain.invoke({
            "chat_history": memory.load_memory_variables({})['history'],
            "input": query
        })

        memory.save_context({"input": query}, {"output": ai_response['answer']})
