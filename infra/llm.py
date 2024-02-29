from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
llm = ChatOpenAI(
    temperature=0.0, 
    verbose=True, 
    model=OPENAI_MODEL,
    streaming=True,
    max_tokens=-1,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)