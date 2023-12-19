from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.llms import Ollama

from dotenv import load_dotenv
import os

load_dotenv()

LLAMACPP_MODEL_PATH = os.environ.get('LLAMACPP_MODEL_PATH')

llm = LlamaCpp(
    model_path=LLAMACPP_MODEL_PATH,
    n_gpu_layers=1,
    n_batch=512,
    # n_ctx=2048,
    max_tokens=-1,
    f16_kv=True,
    temperature=0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# llm = Ollama(
#     model="mistral:latest", 
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     temperature=0, 
# )