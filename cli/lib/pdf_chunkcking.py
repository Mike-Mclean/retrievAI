import numpy as np
import re
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

        doc_chunks = semantic_chunk(text)

        for i, chunk in enumerate(doc_chunks):
            all_chunks.append(chunk)
            metadata.append() 

def semantic_chunk(
        text: str,
        max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    i = 0
    while i < len(sentences):
        if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
            chunk_sentences = sentences[0]
            chunk = chunk_sentences.strip()
        else:
            chunk_sentences = sentences[i : i + max_chunk_size]
            if chunks and len(chunk_sentences) <= overlap:
                break
            chunk = " ".join(chunk_sentences).strip()
        if chunk:
            chunks.append(chunk)
        i += max_chunk_size - overlap

    return chunks

