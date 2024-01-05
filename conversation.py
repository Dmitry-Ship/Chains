from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from utils.llm import llm

DEFAULT_TEMPLATE = """Write a response of the AI. If the AI does not know the answer to a question, it truthfully says it does not know. 

Current conversation:
{history}
Human: {input}
AI:"""
PROMPT = PromptTemplate.from_template(DEFAULT_TEMPLATE)

conversation_chain = ConversationChain(
    prompt=PROMPT,
    llm=llm, 
    memory=ConversationSummaryBufferMemory(llm=llm), 
    verbose=True,
)

while True:
    query = input("\nUser: ")
    conversation_chain.run(input=query)

