import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from support_assistant.vector_store import initialize_vector_store

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

class GraphState(TypedDict):
    question: str
    intent: str
    context: List[str]
    sources: List[str]
    answer: str
    confidence: float

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "pass",
    "tracking", "cancel", "damaged", "spoiled", "gift card",
    "support hours", "chat", "fee"
]

def classify_intent(state: GraphState) -> GraphState:
    q_lower = state["question"].lower()
    if any(keyword in q_lower for keyword in POLICY_KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"
    return {**state, "intent": intent}

def route_intent(state: GraphState) -> str:
    return state["intent"]

def retrieve_and_answer(state: GraphState) -> GraphState:
    collection = initialize_vector_store()
    results = collection.query(
        query_texts=[state["question"]],
        n_results=3
    )
    
    docs = results["documents"][0] if results["documents"] else []
    sources = [m["source"] for m in results["metadatas"][0]] if results["metadatas"] else []
    
    if MOCK_LLM:
        answer = f"Based on Zepto policy, here is the information regarding your request: {docs[0]}" if docs else "No relevant policy found."
        confidence = 0.92
    else:
        answer = docs[0] if docs else "No policy information found."
        confidence = 0.85
        
    return {
        **state,
        "context": docs,
        "sources": sources,
        "answer": answer,
        "confidence": confidence
    }

def direct_answer(state: GraphState) -> GraphState:
    return {
        **state,
        "context": [],
        "sources": [],
        "answer": "Hello! How can I assist you with Zepto services today?",
        "confidence": 1.0
    }

def build_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("policy_question", retrieve_and_answer)
    workflow.add_node("general_question", direct_answer)
    
    workflow.set_entry_point("classify_intent")
    
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "policy_question": "policy_question",
            "general_question": "general_question"
        }
    )
    
    workflow.add_edge("policy_question", END)
    workflow.add_edge("general_question", END)
    
    return workflow.compile()

app_graph = build_workflow()
