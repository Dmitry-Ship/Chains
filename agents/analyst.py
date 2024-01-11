from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import ShellTool
from langchain import hub
from .tools import create_math_tool
from infra.llm import llm
from chains.sql import sql_chain
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

sql_tool = Tool(
    func=lambda x: sql_chain.invoke({"question": x}),
    name="Database",
    description="Use when you need to access database. Only takes a question in natural language as input"
)

tools = [
    ShellTool(),
    create_math_tool(llm),
    sql_tool,
    repl_tool,
]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    handle_parsing_errors=True, 
    max_iterations=10,
    verbose=True,
)

while True:
    query = input("\nTask: ")
    agent_executor.invoke({"input": query})
