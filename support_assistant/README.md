# Zepto Support Assistant Module

An offline-capable RAG support assistant for Zepto built using LangGraph, ChromaDB, and FastAPI.

## Architecture
1. Classification Node: Routes incoming query using heuristic intent classification (policy_question vs general_question).
2. Retrieval Node: Query ChromaDB collection initialized with all-MiniLM-L6-v2 embeddings.
3. Response Output: Pydantic validated JSON response at /ask.

## Run Locally
pip install -r requirements.txt
python -m support_assistant.vector_store
uvicorn support_assistant.main:app --port 8000

## Sample Request
POST /ask
{"question": "What is the return policy for damaged items?"}

## Sample Response
{"answer": "Based on Zepto policy, here is the information regarding your request...", "sources": ["doc_02.txt", "doc_06.txt"], "confidence": 0.92}
