from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from infra.llm import llm
from langchain.tools.retriever import create_retriever_tool
from infra.docs_store import vertor_store

retriever = vertor_store.get_retriever()
tools = [
    create_retriever_tool(
        retriever,
        "Book Children of Ruin",
        "Searches and returns info about the book Children of Ruin",
    )
]

prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        agent_executor.invoke({ "input":query })