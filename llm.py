from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp, Ollama 
from langchain.chat_models import ChatOllama
from langchain.embeddings import LlamaCppEmbeddings, OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL_PATH = os.environ.get('MODEL_PATH')

# llm = LlamaCpp(
#     model_path=MODEL_PATH,
#     n_gpu_layers=1,
#     n_batch=512,
#     n_ctx=4096,
#     max_tokens=-1,
#     f16_kv=True,
#     temperature=0.0,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=True,
#     model_kwargs={'instruct': True, "interactive": True}
# )

OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL')
llm = ChatOllama(
    model=OLLAMA_MODEL, 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.0, 
    verbose=True,
)