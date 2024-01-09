from langchain.chains import ConversationChain
from langchain.memory import  ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from utils.llm import llm

DEFAULT_TEMPLATE = """Write a response of the AI. If the AI does not know the answer to a question, it truthfully says it does not know. 

Current conversation:
{history}
Human: {input}
AI:"""

conversation_chain = ConversationChain(
    prompt=PromptTemplate.from_template(DEFAULT_TEMPLATE),
    memory=ConversationSummaryBufferMemory(llm=llm), 
    llm=llm, 
    verbose=True,
)

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        conversation_chain.run(input=query)

