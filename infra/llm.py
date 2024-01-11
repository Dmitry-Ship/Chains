from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
import os
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL_PATH = os.environ.get('MODEL_PATH')

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=1,
    n_batch=1024,
    n_ctx=8000,
    max_tokens=-1,
    f16_kv=True,
    temperature=0.0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
)


