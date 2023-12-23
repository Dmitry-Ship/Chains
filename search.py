from langchain.prompts import ChatPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from llm import llm

template = """turn the following user input into a search query for a search engine:

{input}"""
prompt_tamplate = ChatPromptTemplate.from_template(template)

search = DuckDuckGoSearchRun()
search_chain = prompt_tamplate | llm | StrOutputParser() | search