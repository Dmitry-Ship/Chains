from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL_PATH = os.environ.get('MODEL_PATH')

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=1,
    n_batch=1024,
    n_ctx=4000,
    max_tokens=-1,
    f16_kv=True,
    temperature=0.0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
)


# llm = OpenAI(
#     base_url="http://localhost:1234/v1", 
#     temperature=0.0, 
#     verbose=True, 
#     streaming=True,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )