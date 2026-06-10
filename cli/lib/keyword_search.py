from collections import defaultdict, Counter
import string
from nltk.stem import PorterStemmer
import pickle
import math

from search_utils import (
    load_stopwords,
    load_parsed_pdfs,
    INDEX_PATH,
    INDEX_DOCMAP_PATH,
    TERM_FREQ_PATH,
    DOC_LENGTHS_PATH,
    BM25_K1,
    BM25_B
    )

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict()

    def __add_document(self, file_name, text):
        tokenized_text = preprocess_text(text)
        self.doc_lengths[file_name] += len(tokenized_text)
        for token in set(tokenized_text):
            self.index[token].add(file_name)
        self.term_frequencies[file_name].update(tokenized_text)

    def build(self) -> None:
        documents = load_parsed_pdfs()
        for doc in documents:
            id = doc["id"]
            self.docmap[id] = doc["file_name"]
            self.__add_document(doc["file_name"], doc["text"])

    def save(self) -> None:
        with open(INDEX_PATH, 'wb') as index_file:
            pickle.dump(self.index, index_file)

        with open(INDEX_DOCMAP_PATH, 'wb') as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open(TERM_FREQ_PATH, 'wb') as term_freq_file:
            pickle.dump(self.term_frequencies, term_freq_file)

        with open(TERM_FREQ_PATH, 'wb') as doc_lengths_file:
            pickle.dump(self.doc_lengths, doc_lengths_file)

    def load(self) -> None:
        try:
            with open(INDEX_PATH, "rb") as index_file:
                self.index = pickle.load(index_file)
        except FileNotFoundError:
            print("Error: index file not found")

        try:
            with open(INDEX_DOCMAP_PATH, "rb") as docmap_file:
                self.docmap = pickle.load(docmap_file)
        except FileNotFoundError:
            print("Error: docmap file not found")

        try:
            with open(TERM_FREQ_PATH, "rb") as term_frq_file:
                self.term_frequencies = pickle.load(term_frq_file)
        except FileNotFoundError:
            print("Error: term frequency file not found")

        try:
            with open(DOC_LENGTHS_PATH, "rb") as doc_lengths_file:
                self.doc_lengths = pickle.load(doc_lengths_file)
        except FileNotFoundError:
            print("Error: document lengths file not found")

    def get_tf(self, file_name: str, term: str) -> int:
        tokenized_term = preprocess_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Error: term is greater than one word")

        processed_term = tokenized_term[0]
        return self.term_frequencies[file_name][processed_term]

    def get_idf(self, term: str) -> float:
        tokenized_term = preprocess_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Error: term is greater than one word")

        token = tokenized_term[0]
        total_pages = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((total_pages + 1) / (term_doc_count + 1))

    def get_tf_idf(self, file_name: str, term: str) -> float:
        return self.get_tf(file_name, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = preprocess_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Error: term is greater than one word")

        token = tokenized_term[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, file_name: str, term: str, k1: float = BM25_K1, b: int = BM25_B) -> float:
        basic_tf = self.get_tf(file_name, term)
        doc_length = self.doc_lengths[file_name]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (basic_tf * (k1 + 1)) / (basic_tf + k1 * length_norm)

    def bm25(self, file_name: str, term: str) -> float:
        bm25_tf = self.get_bm25_tf(file_name, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        tokenized_query = preprocess_text(query)

        scores = defaultdict(float)
        for doc_id in self.docmap.keys():
            for token in tokenized_query:
                scores[doc_id] += self.bm25(doc_id, token)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            pdf_title = self.docmap[doc_id]
            results.append(KeyWordSearchResult(doc_id, pdf_title, score))

        return results

class KeyWordSearchResult:
    def __init__(self, doc_id: str, pdf_title: str, bm25_score: float):
        self.doc_id = doc_id
        self.pdf_title = pdf_title
        self.bm25_score = bm25_score

    def __repr__(self):
        return f"""
        Doc Title: {self.pdf_title}
        Similarity Score: {self.similarity_score}
        Keyword Search score: {self.bm25_score}
    """

def preprocess_text(text: str) -> str:
    text = text.lower()
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)

    stop_words = load_stopwords()
    split_text = text.split()
    split_text = [word for word in split_text if word != '' and word not in stop_words]

    stemmer = PorterStemmer()

    split_text = [stemmer.stem(word) for word in split_text]

    return split_text
