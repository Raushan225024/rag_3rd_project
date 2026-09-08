from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import config.config as config

embedding = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL
)

def retrieve_documents( query, k=6):

    vector_store = QdrantVectorStore.from_existing_collection(
        path=config.VECTOR_DB_DIR,
        collection_name=config.COLLECTION_NAME,
        embedding=embedding,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    return docs