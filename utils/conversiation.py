from llm import llm
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder


template = """
The following is a conversation between a human and an AI. If the AI does not know the answer to a question, it truthfully says it does not know.
Write a single AI response.

Current conversation:
{history}

Human: {input}
AI:"""


PROMPT = PromptTemplate.from_template(template=template)

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm, 
    memory=ConversationSummaryMemory(llm=llm, memory_key="history"), 
    verbose=True,
)