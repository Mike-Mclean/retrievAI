import os

from .keyword_search import InvertedIndex, KeyWordSearchResult
from .pdf_semantic_search import PdfSemanticSearch
from search_utils import (
    load_parsed_pdfs,
    INDEX_PATH,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    DEFAULT_K
)

class HybridSearchResults:
    def __init__(self, doc_id: str,
                 doc_title: str,
                 hybrid_search_score: float,
                 keyword_score: float,
                 semantic_score: float):
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.hybrid_search_score = hybrid_search_score
        self.keyword_score = keyword_score
        self.semantic_score = semantic_score

    def __repr__(self):
        return f"""
        Doc_ID: {self.doc_id}
        Doc Title: {self.doc_title}
        Hybrid Search Score: {self.hybrid_search_score}
        Keyword Score: {self.keyword_score}
        Semantic Score: {self.semantic_score}
    """

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_serach = PdfSemanticSearch()
        self.semantic_serach.load_or_create_chunk_embeddings(documents)
        self.index = InvertedIndex()

        if not os.path.exists(INDEX_PATH):
            self.index.build()
            self.index.save()
        else:
            self.index.load()

    def _BM25_search(self, query, limit = DEFAULT_SEARCH_LIMIT) -> list[KeyWordSearchResult]:
        return self.index.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit = DEFAULT_SEARCH_LIMIT):
        # add type hints
        pass

