from fastapi import FastAPI
from pydantic import BaseModel

from retriever.retriever import retrieve_documents
from llmcall.llm import ask_llm


app = FastAPI(title="RAG API")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question

    # Retrieve relevant documents
    docs = retrieve_documents(question)

    # Send question + documents to LLM
    answer = ask_llm(question, docs)

    return {
        "question": question,
        "answer": answer
    }