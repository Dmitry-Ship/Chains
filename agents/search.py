from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain import hub
from langchain.tools import Tool
from langchain.chains import LLMMathChain
from infra.llm import llm

search = DuckDuckGoSearchRun()
llm_math_chain = LLMMathChain.from_llm(llm=llm)
tools = [
    search, 
    Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you are asked to perform math calculations"
    ),
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


