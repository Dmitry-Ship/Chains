from langchain.agents import create_react_agent, AgentExecutor
import utils.tools as tools
from utils.llm import llm
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain import hub

tools = [
    DuckDuckGoSearchRun(), 
    tools.image_describer,
    tools.create_math(llm),
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


