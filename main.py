from summary import pros_cons_generator
from coder import coder
from facts_extractor import extract_facts_from_url
from search import search_chain
from suggester import suggester
from story import img2speech
from conversiation import conversation
from rag import rag_chain


while True:
    query = input("\nUser: ")
    rag_chain.invoke({"question": query, "system_message": "You are Seknovi" })

    # conversation.predict(input=query)
    # extract_facts_from_url("https://www.theguardian.com/science/2018/may/30/speculative-biology-understanding-the-past-and-predicting-our-future")
    # coder.invoke({"input": "is bird a palindrome?"})
    # print(searcher.invoke({"input": "I'd like to figure out what games are tonight in NYC"}))
    # suggester.invoke('What are your favorite movie genres?')

    # img2speach('https://images.photowall.com/products/69237/lion-close-up.jpg?h=699&q=85')
    # pros_cons_generator.invoke({"input": query}) 