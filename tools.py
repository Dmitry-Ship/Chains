import requests
from llm import llm
from bs4 import BeautifulSoup
from langchain.tools import Tool, DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.story import story_chain, tex2speech, img2text
from utils.pros_and_cons import pros_cons_generator
from utils.rag import rag_chain

output_parser = StrOutputParser()

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

summarize_chain = (
    PromptTemplate.from_template("Summarize the following content: {content}") 
    | llm 
    | output_parser
)

summarizer = Tool(
    func=lambda x: summarize_chain.invoke({"content": x}),
    name="Summarizer",
    description="Summarizes big texts"
)

facts_chain = (
    PromptTemplate.from_template("""
        Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences.
        text: {text_input}
        facts: 
    """)   
    | llm 
    | output_parser
)

facts_extractor = Tool(
    func=lambda x: facts_chain.invoke({"text_input": x}),
    name="FactsExtractor",
    description="Extracts facts from a text"
)

pros_cons_generator = Tool(
    func=lambda x: pros_cons_generator.invoke({"input": x}),
    name="ProsCons",
    description="Generates pros and cons of a given topic"
)

story_teller = Tool(
    func=lambda x: story_chain.invoke({"context": x}),
    name="StoryTeller",
    description="Creates a story based on short context"
)

speaker = Tool(
    func=tex2speech,
    name="Speaker",
    description="Transforms text to speech and saves it to a file"
)

image_describer = Tool(
    func=img2text,
    name="ImageDescriber",
    description="Describes an image from URL"
)


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_search = Tool(
    func=wikipedia.run,
    name="Wikipedia Search",
    description="Useful when users request biographies or historical moments."
)

children_of_time_retriever = Tool(
    func=lambda x: rag_chain.invoke({
        "question": x, 
        "system_message": "Act like Senkovi" 
    }),
    name="Children Of Time",
    description="Use for information related to Children of Time"
)