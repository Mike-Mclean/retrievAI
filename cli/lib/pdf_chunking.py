import numpy as np
import re
import json
import os
from sentence_transformers import SentenceTransformer
from .search_utils import (
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

def detect_chunk_boundaries(text):
    boundaries = []
    patterns =[
        # Numbered headings: "1.", "1.1", "A.", "IV."
        (r"(?m)^(?:[A-Z]{1,4}\.|[IVXLC]+\.|[\d]+(?:\.[\d]+)*\.?)\s+[A-Z]", "numbered_heading"),
        # ALL CAPS lines (section titles, exhibit headers)
        (r"(?m)^[A-Z][A-Z\s\d,.\-:]{4,}$", "caps_heading"),
        # Title case short lines (likely headings, not sentences)
        (r"(?m)^(?:[A-Z][a-z]+\s){1,6}$", "title_case_heading"),
        # "TERM" means / "TERM" shall mean (definition blocks)
        (r'(?m)^\"[A-Z][^\"]+\"\s+(?:means|shall mean)', "definition")
    ]

    for pattern, label in patterns:
        for m in re.finditer(pattern, text):
            boundaries.append((m.start(), label))

    boundaries.sort(key=lambda x: x[0])
    deduped = []
    last_pos = -50
    for pos, label in boundaries:
        if pos - last_pos > 50:
            deduped.append((pos, label))
            last_pos = pos

    return deduped

def split_into_chunks(text, max_size=1500, min_size=100):
    boundaries = detect_chunk_boundaries(text)

    if not boundaries:
        # No structure detected — fall back to paragraph splitting
        return chunk_by_paragraphs(text, max_size)

    # Cut text at each boundary
    cut_points = [0] + [pos for pos, _ in boundaries] + [len(text)]
    raw_chunks = [text[cut_points[i]:cut_points[i+1]].strip()
                  for i in range(len(cut_points) - 1)]
    raw_chunks = [c for c in raw_chunks if c]

    return merge_and_split(raw_chunks, max_size, min_size)

def merge_and_split(chunks, max_size, min_size):
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

    # Split oversized chunks on sentence boundaries
    result = []
    for chunk in merged:
        if len(chunk) <= max_size:
            result.append(chunk)
        else:
            result.extend(split_on_sentences(chunk, max_size))

    return result

def split_on_sentences(text, max_size):
    """Last resort: split a large chunk into sentence-sized pieces."""
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
    return merge_and_split([p.strip() for p in paragraphs if p.strip()], max_size, min_size=100)

