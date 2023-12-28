from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory
from llm import get_llm

llm = get_llm()

conversation_chain = ConversationChain(
    # prompt=PromptTemplate.from_template(template=template),
    llm=llm, 
    memory=ConversationSummaryBufferMemory(llm=llm), 
    verbose=True,
)

while True:
    query = input("\nUser: ")
    conversation_chain.run(input=query)
