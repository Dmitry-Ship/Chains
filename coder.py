from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.utilities import PythonREPL
from llm import llm

template = """Write some python code to solve the user's problem. 

Return a single block of python code in Markdown format and nothing else, e.g.:

```python
....
```"""
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]

coder = prompt | llm | StrOutputParser() | _sanitize_output | PythonREPL().run
