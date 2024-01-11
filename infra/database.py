import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv(override=True)

DB_URI = os.environ.get('DB_URI')

db = SQLDatabase.from_uri(DB_URI, sample_rows_in_table_info=0)