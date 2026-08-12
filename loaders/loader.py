import config.config as config
from langchain_community.document_loaders import PyPDFLoader


def load_documents():
    config.documents = []

    # Load all PDF files from the data folder
    for pdf_file in config.DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        config.documents.extend(loader.load())

    return config.documents