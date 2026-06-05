import os
from pathlib import Path
import json

DEFAULT_MAX_SEMANTIC_CHUNK_SIZE = 1500
DEFAULT_MIN_SEMANTIC_CHUNK_SIZE = 100
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEARCH_LIMIT = 10

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PDF_TESTING_PATH = os.path.join(ROOT_PATH, "data", "pdf_testing")
CACHE_PATH = os.path.join(ROOT_PATH, "cache", "cache.json")
CHUNK_EMBEDDINGS_PATH = os.path.join(ROOT_PATH, "cache","chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(ROOT_PATH, "cache","chunk_metadata.json")
STOPWORDS_PATH = os.path.join(ROOT_PATH, "data", "stopwords.txt")

def list_pdf_documents(pdfs_path: str | Path | os.PathLike = PDF_TESTING_PATH):
    return os.listdir(pdfs_path)

def load_parsed_pdfs(parsed_pdfs_path: str | Path | os.PathLike =  CACHE_PATH) -> list[dict]:
    with open(parsed_pdfs_path, "r") as file:
        pdf_data = json.load(file)
    return pdf_data

def load_stopwords(stopwords_path: str | Path | os.PathLike =  STOPWORDS_PATH) -> list[str]:
    with open(STOPWORDS_PATH, "r") as stopwords_file:
        return stopwords_file.read().splitlines()
