# Chains

setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate && \
pip install -r ./requirements.txt
```

### Chatbot with memory

```bash
python ./coversation.py
```

### Chatbot with RAG

write docs to database

```bash
python ./utils/rag.py <path to docs>
```

run bot without memory

```bash
python ./rag.py
```

run bot with memory

```bash
python ./rag_memory.py
```

### Agent

```bash
python ./agent.py
```
