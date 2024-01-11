from langchain.tools import Tool
from langchain.chains import LLMMathChain

def create_math_tool(llm):
    llm_math_chain = LLMMathChain.from_llm(llm=llm)
 
    return Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you are asked to perform math calculations"
    )



