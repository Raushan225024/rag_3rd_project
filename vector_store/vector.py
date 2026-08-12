from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import config.config as config


def store_vectors():

    if config.embedding_model is None:
        raise ValueError("Embedding model not loaded.")

    if config.chunks is None:
        raise ValueError("Chunks not found.")

    #client = QdrantClient(path=str(config.QDRANT_PATH))

    vector_store = QdrantVectorStore.from_documents(
        documents=config.chunks,
        embedding=config.embedding_model,
        path=str(config.VECTOR_DB_DIR),
        collection_name=config.COLLECTION_NAME,
    )

    return vector_store