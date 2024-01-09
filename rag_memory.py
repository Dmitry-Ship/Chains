from utils.rag import retriever
from utils.llm import llm
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def create_rag_chain_with_memory(llm, retriever):
    condense_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                """Given a chat history and the latest user question \
                which might reference the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
    condense_q_chain = condense_q_prompt | llm 

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer concise.\

            {context}"""),
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


rag_chain_with_memory = create_rag_chain_with_memory(llm, retriever)
chat_history = []

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        ai_msg = rag_chain_with_memory.invoke({
            "question": query, 
            "chat_history": chat_history
        })

        chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg)])
