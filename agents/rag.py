# from langchain import hub
# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from infra.llm import llm
# from rag.docs_store import vector_store

# retriever_tool = create_retriever_tool(
#     vector_store.get_retriever(),
#     "blindsight_search",
#     "Search for information about Blindsight.",
# )

# search = DuckDuckGoSearchRun()
# tools = [search, retriever_tool]

# prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt)
# agent = create_openai_functions_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# # message_history = ChatMessageHistory()
# # agent_with_chat_history = RunnableWithMessageHistory(
# #     agent_executor,
# #     # This is needed because in most real world scenarios, a session id is needed
# #     # It isn't really used here because we are using a simple in memory ChatMessageHistory
# #     lambda session_id: message_history,
# #     input_messages_key="input",
# #     history_messages_key="chat_history",
# # )

# if __name__ == "__main__":
#     while True:
#         agent_executor.invoke(
#             {"input": 'what is the weather like today in SF?'},
#             # This is needed because in most real world scenarios, a session id is needed
#             # It isn't really used here because we are using a simple in memory ChatMessageHistory
#             # config={"configurable": {"session_id": "<foo>"}},
#         )
#         query = input("\nHuman: ")
#         agent_executor.invoke(
#             {"input": query},
#             # This is needed because in most real world scenarios, a session id is needed
#             # It isn't really used here because we are using a simple in memory ChatMessageHistory
#             # config={"configurable": {"session_id": "<foo>"}},
#         )

# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]
completion = client.chat.completions.create(
  model="local-model", # this field is currently unused
  messages=[
    {"role": "system", "content": "You are a helpful ai assistant."},
    {"role": "user", "content": "What's the status of my transaction?."}
  ],
  temperature=0.0,
  tool_choice="auto",
  tools=tools
)

print(completion.choices[0].message)