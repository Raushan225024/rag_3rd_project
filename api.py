from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from retriever.retriever import (
    retrieve_documents,
    load_embedding_model
)

from llmcall.llm import ask_llm


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting RAG API...")

    # Load embedding model only once
    load_embedding_model()

    print("RAG API startup completed")

    yield

    print("RAG API shutting down")


app = FastAPI(
    title="RAG API",
    description="API for asking questions from your RAG system",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def home():
    return {
        "message": "RAG API is running successfully"
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    question = request.question

    docs = retrieve_documents(question)

    answer = ask_llm(question, docs)

    return {
        "question": question,
        "answer": answer
    }