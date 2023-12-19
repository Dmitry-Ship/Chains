from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import llm


output_parser = StrOutputParser()

facts_chain = (
    PromptTemplate.from_template("""
        Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. text: \n\n {text_input}
    """) 
    | llm 
    | output_parser
)

movie_chain = (
    PromptTemplate.from_template("""
        Given these facts come up with a movie idea. facts: \n\n {facts}
    """) 
    | llm 
    | output_parser
)

movie_generator = {"facts": facts_chain} | movie_chain 




