from langchain_huggingface import HuggingFaceEmbeddings
import config.config as config


def load_embedding_model():
    """
    Load the HuggingFace embedding model.
    """

    config.embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    return config.embedding_model


def create_embeddings():
    """
    Create embeddings for all chunks.
    """

    if config.chunks is None:
        raise ValueError("Chunks not found. Create chunks first.")

    if config.embedding_model is None:
        load_embedding_model()

    texts = [chunk.page_content for chunk in config.chunks]

    config.vectors = config.embedding_model.embed_documents(texts)

    return config.vectors