from llm import llm
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder


DEFAULT_TEMPLATE = """
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: 
{input} 
AI: """
PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)


# USER_NAME = "Human"
# AI_NAME = "AI"
# prompt = ChatPromptTemplate(
#     messages=[
#         # The system prompt is now sent directly to llama instead of putting it here
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("\n{user_input}\n" + AI_NAME + ":"),
#         # 
#     ]

# )

# memory = ConversationSummaryBufferMemory(
#         human_prefix=USER_NAME,
#         ai_prefix=AI_NAME,
#         llm=llm,
#         memory_key="chat_history", 
#         return_messages=True, 
#         max_token_limit=7500)

# conversation = ConversationChain(
#     prompt=prompt,
#     input_key="user_input",
#     llm=llm,
#     verbose=True,
#     memory=memory,
# )

conversation = ConversationChain(
    # prompt=PROMPT,
    llm=llm, 
    memory=ConversationSummaryBufferMemory(llm=llm), 
    verbose=True,
)