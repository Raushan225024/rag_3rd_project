from langchain_text_splitters import RecursiveCharacterTextSplitter
import config.config as config


def create_chunks():
    # Check if documents are loaded
    if config.documents is None:
        raise ValueError("No documents found. Please load the documents first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    # Create chunks
    config.chunks = splitter.split_documents(config.documents)

    return config.chunks