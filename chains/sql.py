import re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from infra.llm import llm
from infra.database import db

def get_schema(_):
    table_info = db.get_table_info()
    table_info = re.sub(re.compile("/\*.*?\*/",re.DOTALL) ,"",table_info) # remove all occurrences streamed comments (/*COMMENT */) from string
    return table_info

def run_query(query):
    print('\n💿: running query...')
    return db.run(query)

template = PromptTemplate.from_template("""
Based on the table schema below, write a raw SQL query that would answer the user's question, don't write explanation:
{schema}
                                        
### examples ###
Question: How many boxes are there?
Raw SQL Query: SELECT COUNT(*) FROM boxes;

Question: {question}
Raw SQL Query:
""")
sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | template
    | llm.bind(stop=["Answer:", "Question:"])
)

response_template = PromptTemplate.from_template("""Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}
Answer:""")

sql_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: run_query(x["query"]),
    )
    | response_template
    | llm
)

if __name__ == "__main__":
    while True:
        query = input("\n🤪: ")
        sql_chain.invoke({
            "question": query, 
        })