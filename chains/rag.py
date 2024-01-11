from operator import itemgetter
from infra.vector_store import retriever
from infra.llm import llm
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

ANSWER_PROMPT = ChatPromptTemplate.from_template(
"""Answer the question based only on the following context:
{context}

Question: {question}
""")

conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | llm

chat_history = []

if __name__ == "__main__":
    while True:
        query = input("\nHuman: ")

        ai_msg = conversational_qa_chain.invoke({
            "question": query,
            "chat_history": chat_history,
        })

        chat_history.extend([
            SystemMessage(content='You are Avrana Kern. Act like her.'),
            HumanMessage(content=query), 
            AIMessage(content=ai_msg)
        ])
