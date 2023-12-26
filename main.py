from utils.conversiation import create_conversation_chain
from utils.rag import create_rag_chain
from agents import create_researcher_agent
from llm import get_llm


# import langchain 
# langchain.debug = True

llm = get_llm()
conversation_chain = create_conversation_chain(llm)
rag_chain = create_rag_chain(llm)
researcher_agent = create_researcher_agent(llm)

while True:
    query = input("\nUser: ")
    rag_chain.invoke({
        "question": query, 
        "system_message": "Always answer with a numbered list of facts" 
    })
    # researcher_agent.run(query)
    # conversation_chain.run(input=query)
