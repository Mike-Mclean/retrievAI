import os

DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PDF_TESTING_PATH = os.path.join(ROOT_PATH, "data", "pdf_testing")
CACHE_PATH = os.path.join(ROOT_PATH, "cache", "cache.json")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")

def list_pdf_documents(pdf_path):
    return os.listdir(pdf_path)