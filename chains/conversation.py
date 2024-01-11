from langchain.chains import ConversationChain
from infra.llm import llm

conversation_chain = ConversationChain(llm=llm)

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")
        conversation_chain.run(input=query, stop=["Human:"])

