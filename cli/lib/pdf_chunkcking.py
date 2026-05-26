import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from search_utils import DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class PdfChunkSearch:
    def __init__(self, model_name = 'all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.documents = None
        self.document_map = {}
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.array:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        metadata = []

        for idx, doc in enumerate(documents):
            text = doc.get("text", "")
            if not text.strip():
                continue



def semantic_chunk(
        text: str,
        max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text.strip()
    if not text:
        return []

