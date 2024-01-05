from langchain.agents import initialize_agent, AgentType
import utils.tools as tools
from utils.llm import llm
from utils.rag import retriever
from langchain.tools import DuckDuckGoSearchResults, ShellTool

researcher_agent = initialize_agent(
    tools=[
        DuckDuckGoSearchResults(), 
        ShellTool(),
        tools.wikipedia_search,
        tools.create_knowledge_base(llm, retriever), 
        tools.create_llm_tool(llm),
        tools.speaker,
        tools.image_describer,
        tools.create_math(llm)
    ], 
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True,
    # max_iterations=2,
)

while True:
    query = input("\nUser: ")
    researcher_agent.run(query)
