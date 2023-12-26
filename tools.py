import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool, DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
        description="use this for general purpose queries and logic"
    )

ddg_search = DuckDuckGoSearchResults()

def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
    })
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()


web_fetcher = Tool(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

speaker = Tool(
    func=tex2speech,
    name="Speaker",
    description="Use it to save text to an audio file. The input to this tool should be the full text."
)

image_describer = Tool(
    func=img2text,
    name="ImageDescriber",
    description="Use it to create describes an image from URL"
)

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_search = Tool(
    func=wikipedia.run,
    name="Wikipedia Search",
    description="Useful when users request biographies or historical moments."
)

def create_knowledge_base(llm):
    return Tool(
        func=lambda x, y: create_rag_chain(llm).invoke({
            "question": x, 
            "system_message": y 
        }),
        name="Children of Ruin Knowledge Base",
        description="Use it for queries related to Children of Ruin."
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
        description="Use to turn a product idea into detailed requirements for a developer. The input to this tool should be the idea." 
    )