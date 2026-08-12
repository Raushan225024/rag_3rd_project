import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY from the given context.

If the answer is not present in the context, say:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


def ask_llm(question, docs):

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response.content