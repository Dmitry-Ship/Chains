from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import ShellTool
from langchain import hub
from infra.llm import llm

tools = [
    ShellTool(),
]

prompt = hub.pull("hwchase17/react")
print(prompt)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    handle_parsing_errors=True, 
    max_iterations=10
)

while True:
    query = input("\nTask: ")
    agent_executor.invoke({"input": query})
