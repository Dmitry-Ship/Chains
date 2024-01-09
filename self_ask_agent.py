from langchain.agents import create_self_ask_with_search_agent, AgentExecutor
import utils.tools as tools
from utils.llm import llm
from langchain import hub
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun


tools = [
    DuckDuckGoSearchRun(max_results=1, name="Intermediate Answer"), 
]

prompt = hub.pull("hwchase17/self-ask-with-search")
agent_executor = AgentExecutor(
    agent=create_self_ask_with_search_agent(llm=llm, tools=tools, prompt=prompt), 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True, 
    max_iterations=10
)

while True:
    query = input("\nTask: ")
    agent_executor.invoke({"input": query})


