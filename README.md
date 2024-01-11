# Chains

setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate && \
pip install -r ./requirements.txt
```

### Chatbot

```bash
python -m chains.conversation
```

### RAG

write docs to database

```bash
python ./infra/vectore_store.py <path to docs>
```

chat

```bash
python -m chains.rag
```

run bot with memory

### Agents

researcher

```bash
python -m agents.researcher
```

developer

```bash
python -m agents.developer
```
