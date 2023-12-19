from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from operator import itemgetter
from llm import llm

output_parser = StrOutputParser()

chain_one = (
    ChatPromptTemplate.from_template('''You are a movie genre survey. Ask the user about their favorite genres.
    Ask:''')
    | llm
    | output_parser
)

chain_two = (
    ChatPromptTemplate.from_template('''You are a movie recommender. Given the user's favorite genres: {genres}, suggest some movies that fall under these genres.
 
    Suggest movies:''')
    | llm
    | output_parser
)

# simplifier = ( 
#     ChatPromptTemplate.from_template(
#         "Turn the concept description of {concept} and explain it to me like I'm five in 500 words",
#     )
#     | llm
#     | output_parser
# )

suggester = chain_one |{"genres": RunnablePassthrough()}| chain_two

