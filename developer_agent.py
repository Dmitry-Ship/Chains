from langchain.agents import create_react_agent, AgentExecutor
import utils.tools as tools
from utils.llm import llm
from langchain.tools import DuckDuckGoSearchResults, ShellTool
from langchain import hub

tools = [
    DuckDuckGoSearchResults(), 
    ShellTool(),
    tools.create_math(llm),
    tools.repl_tool,
    tools.create_pm(llm),
    tools.sql_tool,
]

prompt = hub.pull("hwchase17/react")
agent_executor = AgentExecutor(
    agent=create_react_agent(llm=llm, tools=tools, prompt=prompt), 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True, 
    max_iterations=10
)

while True:
    query = input("\nTask: ")
    agent_executor.invoke({"input": query})
