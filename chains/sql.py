from infra.llm import llm
from infra.database import db
from langchain_experimental.sql import SQLDatabaseChain

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

if __name__ == "__main__":
    while True:
        query = input("\n🤪: ")
        db_chain.invoke({
            "query": query, 
        })