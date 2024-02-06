from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp, Ollama
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

MODEL_PATH = os.environ.get('MODEL_PATH')

# llm = LlamaCpp(
#     model_path=MODEL_PATH,
#     n_gpu_layers=1,
#     n_batch=1024,
#     n_ctx=4000,
#     max_tokens=-1,
#     f16_kv=True,
#     temperature=0.0,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=False,
# )


# llm = OpenAI(
#     temperature=0.0, 
#     verbose=False, 
#     streaming=True,
#     max_tokens=-1,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL')
llm = Ollama(
    model=OLLAMA_MODEL,
    temperature=0.0, 
    verbose=False, 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)