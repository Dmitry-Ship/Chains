from utils.conversiation import conversation
from utils.rag import rag_chain
from agents import reasearcher_agent, story_teller_agent

# conversation.run(input="what is my name?")
# conversation.run(input="my name Dima")
# conversation.run(input="what is my name?")

while True:
    query = input("\nUser: ")
    # rag_chain.invoke({
    #     "question": query, 
    #     "system_message": "Act like Senkovi" 
    # })
    # story_teller_agent.invoke(query) 
    # reasearcher_agent.invoke(query)
    # conversation.run(input=query)
