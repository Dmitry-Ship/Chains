from langchain.agents import initialize_agent, AgentType
from llm import llm
import tools
from langchain.memory import ConversationKGMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory

reasearcher_agent = initialize_agent(
    tools=[tools.ddg_search, tools.web_fetcher, tools.facts_extractor, tools.wikipedia_search], 
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
)

story_teller_agent = initialize_agent(
    tools=[tools.story_teller, tools.speaker, tools.image_describer], 
    llm=llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
)

state_of_the_art_agent = initialize_agent(
    tools=[tools.children_of_time_retriever, tools.ddg_search], 
    llm=llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    memory=ConversationSummaryMemory(llm=llm), 
)