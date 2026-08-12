from retriever.retriever import retrieve_documents
from llmcall.llm import ask_llm

print("=" * 60)
print("RAG Chat")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    docs = retrieve_documents(question)

    answer = ask_llm(question, docs)

    print("\nAssistant:")
    print(answer)