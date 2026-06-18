import os

from .keyword_search import InvertedIndex, KeyWordSearchResult
from .pdf_semantic_search import PdfSemanticSearch, RetrievalResult
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

    def normalize_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)
        if minimum == maximum:
            return [1.0] * len(scores)

        normalized_scores = []
        for score in scores:
            norm_score = (score - minimum) / (maximum - minimum)
            normalized_scores.append(norm_score)

        return normalized_scores

    def hybrid_score(self, bm25_score, semantic_score, alpha = DEFAULT_ALPHA) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def normalize_search_results(self, results: list[KeyWordSearchResult] | list[RetrievalResult]) -> list[HybridSearchResults]:
        scores = []
        for result in results:
            scores.append(result.hybrid_search_score)

        normalized = self.normalize_scores(scores)
        for i, result in enumerate(results):
            result.normalized_score = normalized[i]

        return results

    def combine_search_results(self, combined_search_results: dict[dict], search_results: dict[HybridSearchResults]) -> None:
        for result in search_results:
            doc_id = result.doc_id
            if doc_id not in combined_search_results:
                combined_search_results[doc_id] = {
                    "title": result.doc_title,
                    "keyword_score": 0.0,
                    "semantic_score": 0.0
                }
            if result.normalized_score > combined_search_results[doc_id]["keyword_score"]:
                combined_search_results[doc_id]["keyword_score"] = result.normalized_score

    def weighted_search(self, query, alpha, limit = DEFAULT_SEARCH_LIMIT):
        keyword_search_results = self._BM25_search(query, limit * 500)
        normalized_keyword_search = self.normalize_search_results(keyword_search_results)

        semantic_search_results = self.semantic_serach.search_chunks(query, limit * 500)
        normalized_semantic_search = self.normalize_search_results(semantic_search_results)

        combined_search_results = {}
        self.combine_search_results(combined_search_results, normalized_keyword_search)
        self.combine_search_results(combined_search_results, normalized_semantic_search)

        hybrid_search_results = []
        for doc_id, data in combined_search_results.items():
            score = self.hybrid_score(data["keyword_score"], data["semantic_score"], alpha)
            hybrid_search_results.append(
                HybridSearchResults(
                    doc_id= doc_id,
                    doc_title= data["title"],
                    hybrid_search_score= score,
                    keyword_score= data["keyword_score"],
                    semantic_score= data["semantic_score"]
                )
            )

        return sorted(hybrid_search_results, key= lambda x: x.score, reverse=True)



