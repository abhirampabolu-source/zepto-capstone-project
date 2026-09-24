from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from support_assistant.graph import app_graph

app = FastAPI(title="Zepto Support Assistant", version="1.0")

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the delivery fee for orders under INR 149?")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "question": request.question,
        "intent": "",
        "context": [],
        "sources": [],
        "answer": "",
        "confidence": 0.0
    }
    
    result = app_graph.invoke(initial_state)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("support_assistant.main:app", host="0.0.0.0", port=8000, reload=True)
