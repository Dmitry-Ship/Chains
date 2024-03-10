from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from infra.llm import llm
from rag.docs_store import blindsight_store

blindsight_retriever = create_retriever_tool(
    blindsight_store.get_retriever(),
    "blindsight_retriever",
    "Search for information from novel Blindsight",
)

search = DuckDuckGoSearchRun()
tools = [search, blindsight_retriever]

prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        agent_with_chat_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": "<foo>"}},
        )

