import numpy as np
import re
import json
import os
from sentence_transformers import SentenceTransformer
from search_utils import (
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    load_parsed_pdfs,
    )

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

        for doc in documents:
            text = doc.get("text", "")

            if not text.strip():
                continue

            doc_chunks = semantic_chunk(text)

            for i, chunk in enumerate(doc_chunks, 1):
                all_chunks.append(chunk)
                metadata.append({
                    "document_id": doc["id"],
                    "document_title": doc["file_name"],
                    "page_number": doc["page"],
                    "chunk_id": i,
                    "total_chunks": len(doc_chunks)
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.array:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as meta_data_file:
                data = json.load(meta_data_file)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

def semantic_chunk(
        text: str,
        max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text.strip()
    if not text:
        return []

    text = normalize_pdf_text(text)
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

def normalize_pdf_text(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"(?<![.!?:])\n(?![A-Z\d])", " ", text)
    text = re.sub(r"\n(?=[A-Z\d])", "\n", text)
    return text.strip()

def try_pdf_chunking():
    pdf_chunk_search = PdfChunkSearch()
    pdf_data = load_parsed_pdfs()
    return pdf_chunk_search.load_or_create_chunk_embeddings(pdf_data)

if __name__ == "__main__":
    try_pdf_chunking()