from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory


def create_conversation_chain(llm):
    # template = """
    # The following is a conversation between a human and an AI. 
    # The AI provides specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    # Write a single response for AI.

    # Current conversation:
    # {history}
    # Human: {input}
    # AI:"""
    return ConversationChain(
        # prompt=PromptTemplate.from_template(template=template),
        llm=llm, 
        # memory=ConversationBufferMemory(llm=llm, memory_key="history"), 
        memory=ConversationBufferMemory(), 
        verbose=True,
    )