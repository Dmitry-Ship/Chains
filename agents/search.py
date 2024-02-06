from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain import hub
from .tools import create_math_tool
from infra.llm import llm

search = DuckDuckGoSearchRun(max_results=1)
tools = [
    search, 
    create_math_tool(llm),
]

prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False,
    handle_parsing_errors=True, 
    max_iterations=10
)

chat_history = ''
while True:
    query = input("\nTask: ")
    result = agent_executor.invoke({"input": query, 'chat_history': chat_history})
    chat_history += f'Human: {query}\nAI: {result["output"]}\n'


