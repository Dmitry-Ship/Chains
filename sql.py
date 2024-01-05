from langchain.prompts import PromptTemplate
from langchain.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
from utils.llm import llm

load_dotenv(override=True)

DB_URI = os.environ.get('DB_URI')

db = SQLDatabase.from_uri(DB_URI)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    print('\n💿: running query...')
    return db.run(query)

template = """Based on the table schema below, write a raw SQL query that would answer the user's question, don't write explanation:
{schema}

Question: {question}
SQL Query:"""
sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | PromptTemplate.from_template(template)
    | llm.bind(stop=["\nSQLResult:", "Answer:", "Explanation:"])
)

response_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: run_query(x["query"]),
    )
    | PromptTemplate.from_template(response_template)
    | llm
)

while True:
    query = input("\n🤪: ")
    full_chain.invoke({
        "question": query, 
    })