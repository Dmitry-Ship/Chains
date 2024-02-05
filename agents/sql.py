import os
from dotenv import load_dotenv
from infra.llm import llm
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

load_dotenv(override=True)
DB_URI = os.environ.get('DB_URI')
db = SQLDatabase.from_uri(DB_URI, sample_rows_in_table_info=0)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

while True:
    query = input("\nTask: ")
    agent_executor.invoke({"input": query})
