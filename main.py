import config.config as config
from loaders.loader import load_documents
from chunks.chunker import create_chunks
import embedding.embedder as embedder
from vector_store.vector import store_vectors
from retriever.retriever import retrieve_documents
from llmcall.llm import ask_llm
print(f"BASE_DIR: {config.BASE_DIR}")
print(f"DATA_DIR: {config.DATA_DIR}")
documents = load_documents()
print(f"Number of documents loaded: {len(documents)}")
chunks = create_chunks()
print(f"Number of chunks created: {len(chunks)}")
print(f"First chunk: {chunks[1].page_content if chunks else 'No chunks created'}")
embedder.load_embedding_model()
config.vectors = embedder.create_embeddings()
print(f"Vectors : {len(config.vectors)}")
print(f"Vector Dimension : {len(config.vectors[0])}")

store = store_vectors()

print("Vectors stored successfully.")



