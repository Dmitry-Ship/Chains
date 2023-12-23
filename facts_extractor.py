from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import llm
from langchain.document_loaders import WebBaseLoader

prompt_template = PromptTemplate.from_template("""
    Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences.
    text: {text_input}
    facts: 
""") 
output_parser = StrOutputParser()

facts_chain = prompt_template | llm | output_parser


def extract_facts_from_url(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    facts_chain.invoke({"text_input": data})