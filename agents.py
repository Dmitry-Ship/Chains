from langchain.agents import initialize_agent, AgentType
import tools
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

def create_researcher_agent(llm):
    return initialize_agent(
        tools=[
            tools.ddg_search, 
            tools.web_fetcher,
            tools.create_pm(llm),
            tools.wikipedia_search,
            tools.create_knowledge_base(llm), 
            tools.create_llm_tool(llm),
            tools.speaker,
            tools.image_describer,
        ], 
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        handle_parsing_errors=True,
        # early_stopping_method="generate",
        # max_iterations=2,
        # memory=ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True), 
    )

# python_agent = create_python_agent(
#     llm=llm,
#     tool=PythonREPLTool(),
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )
