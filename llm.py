from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp, Ollama 
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def get_llm():
    MODEL = os.environ.get('MODEL')
    match MODEL:
        case "ollama":
            print("You can become a web developer.")
            OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL')
            return Ollama(
                model=OLLAMA_MODEL, 
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature=0.0, 
                verbose=True,
            )
        case "llamacpp" |  _:
            LLAMACPP_MODEL_PATH = os.environ.get('LLAMACPP_MODEL_PATH')
            return LlamaCpp(
                model_path=LLAMACPP_MODEL_PATH,
                n_gpu_layers=1,
                n_batch=2024,
                n_ctx=7096,
                max_tokens=-1,
                f16_kv=True,
                temperature=0.0,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=True,
                stop=["Human:"]
                # model_kwargs={'instruct': True, "interactive": True}
            )
