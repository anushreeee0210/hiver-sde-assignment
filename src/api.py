from fastapi import FastAPI
from pydantic import BaseModel

from src.generate_response import generate_response


app = FastAPI(
    title="AppleSupport Customer Support API",
    description="AI-powered customer support reply system using intent detection and retrieval.",
    version="1.0.0"
)


class ReplyRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "AppleSupport Customer Support API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/reply")
def reply(request: ReplyRequest):

    result = generate_response(request.query)

    return {
        "query": request.query,
        "intent": result["intent"],
        "response": result["response"],
        "similarity": round(result["similarity"], 4),
        "retrieved_issue": result["source"]
    }