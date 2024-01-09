from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp, Ollama 
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def get_llm():
    MODEL = os.environ.get('MODELs')
    match MODEL:
        case "ollama":
            OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL')
            return Ollama(
                model=OLLAMA_MODEL, 
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature=0.0,
                verbose=True,
                # stop=["Human:", "Explanation:"] 
            )
        case "llamacpp" |  _:
            LLAMACPP_MODEL_PATH = os.environ.get('LLAMACPP_MODEL_PATH')

            return LlamaCpp(
                model_path=LLAMACPP_MODEL_PATH,
                n_gpu_layers=1,
                n_batch=1024,
                n_ctx=2096,
                max_tokens=-1,
                f16_kv=True,
                temperature=0.0,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=True,
                # stop=["Human:", "Observation:", "Explanation:"]
            )

llm = get_llm()


# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# key_points_prompt = PromptTemplate.from_template("""
# You are a movie script writer. Write key plot points and themes that sequel of the movie {movie} must have. Don't write the actual script.
# Plot points and themes:
# """)

# script_prompt = PromptTemplate.from_template("""
# You are a movie script writer. Write a script for the sequel of the movie {movie}. The script must include the following plot points.
# Plot points: {plot_points}

# Script:
# """)

# sequel_chain = {
#     "movie": RunnablePassthrough(),
#     "plot_points": key_points_prompt | llm,
# } | script_prompt | llm

# # sequel_chain.invoke({
# #     "movie": "inception"
# # })


