from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import ShellTool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from infra.llm import llm
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit

working_directory = TemporaryDirectory()
toolkit = FileManagementToolkit(root_dir=str(working_directory.name)).get_tools()
tools = [ShellTool()] + toolkit

prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False,
    handle_parsing_errors=True, 
    max_iterations=10
)

message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    query = input("\nTask: ")
    agent_with_chat_history.invoke(
        {"input": query},
        config={"configurable": {"session_id": "<foo>"}},
    )
