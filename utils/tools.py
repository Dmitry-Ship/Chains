from langchain.tools import Tool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from utils.story import tex2speech, img2text
from rag_memory import create_rag_chain_with_memory
from sql import full_chain

speaker = Tool(
    func=tex2speech,
    name="Speaker",
    description="Saves text to an audio file. Takes a text as input."
)

image_describer = Tool(
    func=img2text,
    name="ImageDescriber",
    description="Describes an image from URL. Takes a single URL as input."
)

def create_knowledge_base(llm, retriever):
    return Tool(
        func=lambda x: create_rag_chain_with_memory(llm, retriever).invoke({
            "question": x, 
            "system_message": "" 
        }),
        name="Book Children of Ruin",
        description="Use it for queries related to the book Children of Ruin."
    )

def create_pm(llm):
    pm_chain = (
        PromptTemplate.from_template("""
        You are a project manager. You will turn a product idea into detailed technical requirements for a developer.
        idea: {idea}
        requirements:
        """)
        | llm
    )

    overall_chain = (
        {
            "requirements": pm_chain,
        }
        | PromptTemplate.from_template("""
        You are a python developer. Write code that meets the following requirements. The response must be a single code block.
        requirements: {requirements}
        code:
        """)
        | llm 
    )

    return Tool(
        func=lambda x: overall_chain.invoke({"idea": x}),
        name="Project Manager",
        description="Use when you want to develop in IT product. This tool turn the idea into detailed technical requirements for a developer."
    )


def create_math(llm):
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
 
    return Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you are asked to perform math calculations"
    )

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

sql_tool = Tool(
        func=lambda x: full_chain.invoke({"query": x}),
        name="SQL",
        description="Use when you need to access databases"
    )