import os
from pathlib import Path
import json

DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PDF_TESTING_PATH = os.path.join(ROOT_PATH, "data", "pdf_testing")
CACHE_PATH = os.path.join(ROOT_PATH, "cache", "cache.json")
CHUNK_EMBEDDINGS_PATH = os.path.join(ROOT_PATH, "cache","chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(ROOT_PATH, "cache","chunk_metadata.json")

def list_pdf_documents(pdfs_path: str | Path | os.PathLike = PDF_TESTING_PATH):
    return os.listdir(pdfs_path)

def load_parsed_pdfs(parsed_pdfs_path: str | Path | os.PathLike =  CACHE_PATH) -> list[dict]:
    with open(CACHE_PATH, "r") as file:
        pdf_data = json.load(file)
    return pdf_data