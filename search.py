from langchain.prompts import ChatPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from llm import llm

search = DuckDuckGoSearchRun()

template = """turn the following user input into a search query for a search engine:

{input}"""
prompt = ChatPromptTemplate.from_template(template)

searcher = prompt | llm | StrOutputParser() | search