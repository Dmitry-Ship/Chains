# Chains

setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate && \
pip install -r ./requirements.txt
```

### RAG

write docs to database

```bash
python python -m rag.docs_store <path to docs>
```

chat

```bash
python -m rag.docs
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
