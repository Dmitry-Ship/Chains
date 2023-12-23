from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm import llm

output_parser = StrOutputParser()

servey_chain = (
    ChatPromptTemplate.from_template('''You are a movie genre survey. Ask the user about their favorite genres.
    Ask:''')
    | llm
    | output_parser
)

reccomendation_chain = (
    ChatPromptTemplate.from_template('''You are a movie recommender. Given the user's favorite genres: {genres}, suggest some movies that fall under these genres.
 
    Suggest movies:''')
    | llm
    | output_parser
)

suggester = servey_chain |{"genres": RunnablePassthrough()}| reccomendation_chain

