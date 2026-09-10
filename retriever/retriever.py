from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import config.config as config

print("STEP 1: retriever.py imported")

embedding = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL
)

print("STEP 2: embedding model loaded")


def retrieve_documents(query, k=6):

    print("STEP 3: retrieving documents")

    vector_store = QdrantVectorStore.from_existing_collection(
        path=config.VECTOR_DB_DIR,
        collection_name=config.COLLECTION_NAME,
        embedding=embedding,
    )

    print("STEP 4: vector store loaded")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever.invoke(query)