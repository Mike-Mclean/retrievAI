import numpy as np
import re
import json
import os
from sentence_transformers import SentenceTransformer
from .search_utils import (
    DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
    DEFAULT_MIN_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_SEARCH_LIMIT,
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

            doc_chunks = chunk_pdf(text)

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

    def generate_embedding(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Error: text doesn't exist or is only whitespace")
        embedding = self.model.encode([text])
        return embedding[0]

    def search_chunks(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        embedding = self.generate_embedding(query)
        chunk_scores = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "document_id": self.chunk_metadata[i]["document_id"],
                    #Add more data to the chunk scores.
                }
            )

def chunk_pdf(text, max_size=DEFAULT_MAX_SEMANTIC_CHUNK_SIZE, min_size=DEFAULT_MIN_SEMANTIC_CHUNK_SIZE):
    text = normalize_pdf_text(text)
    return split_into_chunks(text, max_size, min_size)


def normalize_pdf_text(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"(?<![.!?:])\n(?![A-Z\d])", " ", text)
    text = re.sub(r"\n(?=[A-Z\d])", "\n", text)
    return text.strip()

def detect_chunk_boundaries(text):
    boundaries = []
    patterns =[
        # Numbered headings: "1.", "1.1", "A.", "IV."
        r"(?m)^(?:[A-Z]{1,4}\.|[IVXLC]+\.|[\d]+(?:\.[\d]+)*\.?)\s+[A-Z]",
        # ALL CAPS lines (section titles, exhibit headers)
        r"(?m)^[A-Z][A-Z\s\d,.\-:]{4,}$",
        # Title case short lines (likely headings, not sentences)
        r"(?m)^(?:[A-Z][a-z]+\s){1,6}$",
        # "TERM" means / "TERM" shall mean (definition blocks)
        r'(?m)^\"[A-Z][^\"]+\"\s+(?:means|shall mean)'
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, text):
            boundaries.append(m.start())

    boundaries.sort(key=lambda x: x[0])

    return boundaries

def split_into_chunks(text,
                      max_size=DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
                      min_size=DEFAULT_MIN_SEMANTIC_CHUNK_SIZE,
                      overlap_size = DEFAULT_CHUNK_OVERLAP):

    boundaries = detect_chunk_boundaries(text)

    if not boundaries:
        return chunk_by_paragraphs(text, max_size)

    split_points = [0]
    for point in boundaries:
        split_points.append(point)
    split_points.append(len(text))

    raw_chunks = []
    for point in range(len(split_points) - 1):
        new_chunk = text[split_points[point] : split_points[point + 1]].strip()
        if new_chunk:
            raw_chunks.append(new_chunk)

    merged = merge_and_split(raw_chunks, max_size, min_size)

    overlapped = []
    for i, chunk in enumerate(merged):
        if i > 0:
            prefix = merged[i - 1][-overlap_size:]
        else:
            prefix = ""

        if i < len(merged) - 1:
            suffix = merged[i + 1][:overlap_size]
        else:
            suffix = ""

        overlapped.append(f"{prefix} {chunk} {suffix}".strip())

    return overlapped

def merge_and_split(chunks,
                    max_size=DEFAULT_MAX_SEMANTIC_CHUNK_SIZE,
                    min_size=DEFAULT_MIN_SEMANTIC_CHUNK_SIZE):
    """Merge undersized chunks and split oversized chunks"""
    merged = []
    buffer = ""
    for chunk in chunks:
        if len(buffer) + len(chunk) < min_size:
            buffer = (buffer + " " + chunk).strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)

    result = []
    for chunk in merged:
        if len(chunk) <= max_size:
            result.append(chunk)
        else:
            result.extend(split_on_sentences(chunk, max_size))

    return result

def split_on_sentences(text, max_size=DEFAULT_MAX_SEMANTIC_CHUNK_SIZE):
    """Split a large chunk into sentence-sized pieces."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    buffer = ""
    for sentence in sentences:
        if len(buffer) + len(sentence) <= max_size:
            buffer = (buffer + " " + sentence).strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks

def chunk_by_paragraphs(text, max_size):
    paragraphs = re.split(r"\n{2,}", text)
    stripped_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return merge_and_split(stripped_paragraphs, max_size)

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
