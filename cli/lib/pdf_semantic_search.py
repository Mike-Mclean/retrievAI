import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_SEARCH_LIMIT,
    load_parsed_pdfs
    )
from .pdf_chunking import (
    chunk_pdf,
    cosine_similarity
)

class PdfSemanticSearch:
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
                    "total_chunks": len(doc_chunks),
                    "chunk_text": chunk
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
            chunk_data = self.chunk_metadata[i]
            chunk_scores.append(
                RetrievalResult(
                    chunk_data["chunk_text"],
                    chunk_data["document_title"],
                    chunk_data["page_number"],
                    similarity,
                    chunk_data["chunk_id"]
                )
            )

        sorted_scores = sorted(chunk_scores, key=lambda result: result.similarity, reverse=True)
        return sorted_scores[:limit]

class RetrievalResult:
    def __init__(self, chunk_text: str, source_pdf: str, page_number: int,
                 similarity_score: float, chunk_id: int):
        self.chunk_text = chunk_text
        self.source_pdf = source_pdf
        self.page_number = page_number
        self.similarity_score = similarity_score
        self.chunk_id = chunk_id

    def __repr__(self):
        return f"""Chunk_text: {self.chunk_text[:100]}\n
        Source_pdf: {self.source_pdf}\n
        Page_number: {self.page_number}\n
        Chunk_similarity: {self.similarity_score}\n
        Chunk_ID: {self.chunk_id}
    """

def retrieve(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[RetrievalResult]:
    documents = load_parsed_pdfs()
    pdf_search = PdfSemanticSearch()
    pdf_search.load_or_create_chunk_embeddings(documents)
    return pdf_search.search_chunks(query, limit)
