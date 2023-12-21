from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import llm

prompt_template = PromptTemplate.from_template("""
    Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences.
    text: {text_input}
    facts: 
""") 
output_parser = StrOutputParser()

facts_chain = prompt_template | llm | output_parser