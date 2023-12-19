from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from llm import llm

output_parser = StrOutputParser()

planner = (
    ChatPromptTemplate.from_template("Generate an argument about: {input}")
    | llm
    | output_parser
    | {"base_response": RunnablePassthrough()}
)

arguments_for = (
    ChatPromptTemplate.from_template(
        "List the pros or positive aspects of {base_response}"
    )
    | llm
    | output_parser
)
arguments_against = (
    ChatPromptTemplate.from_template(
        "List the cons or negative aspects of {base_response}"
    )
    | llm
    | output_parser
)

final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "Pros:\n{results_1}\n\nCons:\n{results_2}"),
            ("system", "Generate a final response given the critique"),
        ]
    )
    | llm
    | output_parser
)

pros_cons_generator = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)
