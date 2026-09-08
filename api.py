from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from retriever.retriever import retrieve_documents
from llmcall.llm import ask_llm


app = FastAPI(
    title="RAG API",
    description="API for asking questions from your RAG system"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QuestionRequest(BaseModel):
    question: str


# Response model
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

    # Retrieve relevant documents
    docs = retrieve_documents(question)

    # Ask LLM using retrieved documents
    answer = ask_llm(question, docs)

    return {
        "question": question,
        "answer": answer
    }