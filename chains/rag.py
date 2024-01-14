from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RagMemoryChain:
    def __init__(self, llm, rag_prompt, memory, retriever):
        self.memory = memory
        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        question_generator_chain = rephrase_prompt | llm | StrOutputParser()
        full_chain = question_generator_chain | qa_chain
        self.full_chain = full_chain

    def invoke(self, query):
        result = self.full_chain.invoke({ 
            "input": query, 
            "chat_history": self.memory.load_memory_variables({})
        })

        self.memory.save_context({"input": query}, {"output": result})
        return result