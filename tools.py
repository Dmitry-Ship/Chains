import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool, DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMMathChain
from langchain_experimental.utilities import PythonREPL
from utils.story import tex2speech, img2text
from utils.rag import create_rag_chain

output_parser = StrOutputParser()

def create_llm_tool(llm):
    llm_chain = (
        PromptTemplate.from_template("{input}")
        | llm
        | output_parser
    )

    return Tool(
        func=lambda x: llm_chain.invoke({"input": x}),
        name="LLM",
        description="Use to for general purpose queries and logic"
    )

ddg_search = DuckDuckGoSearchResults()

def fetch_web_page(url):
    response = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
    })
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()


web_fetcher = Tool(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetching content of a web page. Takes a single URL as input"
)

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

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_search = Tool(
    func=wikipedia.run,
    name="Wikipedia Search",
    description="Use to request biographies or historical moments."
)

def create_knowledge_base(llm, retriever):
    return Tool(
        func=lambda x: create_rag_chain(llm, retriever).invoke({
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
        | output_parser
    )
    return Tool(
        func=lambda x: pm_chain.invoke({"idea": x}),
        name="Project Manager",
        description="Turns a product idea into detailed technical requirements for a developer."
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


from langchain.tools import ShellTool

shell_tool = ShellTool()