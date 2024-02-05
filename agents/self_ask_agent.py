from langchain.agents import create_self_ask_with_search_agent, AgentExecutor
from langchain import hub
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from infra.llm import llm

tools = [
    DuckDuckGoSearchRun(max_results=1, name="Intermediate Answer"), 
]

prompt = hub.pull("hwchase17/self-ask-with-search")
agent_executor = AgentExecutor(
    agent=create_self_ask_with_search_agent(llm=llm, tools=tools, prompt=prompt), 
    tools=tools, 
    handle_parsing_errors=True, 
    max_iterations=10,
    verbose=False,
)

while True:
    query = input("\nTask: ")
    agent_executor.invoke({"input": query, 'stop': ["Explanation:"]})


