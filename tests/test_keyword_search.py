from cli.lib.keyword_search import *
from cli.lib.search_utils import load_parsed_pdfs, INDEX_PATH

import unittest
from pathlib import Path

class TestInvertedIndex(unittest.TestCase):

    def setUp(self):
        self.keyword_search = InvertedIndex()
        self.keyword_search.build()
        self.keyword_search.save()
        self.keyword_search.load()

    def test_building_inverted_index(self):
        self.keyword_search.build()
        index_file = Path(INDEX_PATH)
        self.assertTrue(index_file.is_file())

    def test_save_inverted_index(self):
        self.keyword_search.save()
        index_file = Path(INDEX_PATH)
        self.assertTrue(index_file.is_file())

    def test_load_inverted_index(self):
        self.keyword_search.load()
        index_file = Path(INDEX_PATH)
        self.assertTrue(index_file.is_file())

    def test_get_tf(self):
        file_id = load_parsed_pdfs()[0]["id"]
        term = "Affiliate"
        tf = self.keyword_search.get_tf(file_id, term)
        self.assertIsInstance(tf, int)

    def test_get_idf(self):
        term = "Affiliate"
        idf = self.keyword_search.get_idf(term)
        self.assertIsInstance(idf, float)

    def test_get_tf_idf(self):
        file_id = load_parsed_pdfs()[0]["id"]
        term = "Affiliate"
        tf_idf = self.keyword_search.get_tf_idf(file_id, term)
        self.assertIsInstance(tf_idf, float)

    def test_get_bm25_idf(self):
        term = "Affiliate"
        bm_25_idf = self.keyword_search.get_bm25_idf(term)
        self.assertIsInstance(bm_25_idf, float)

    def test_get_bm25_tf(self):
        file_id = load_parsed_pdfs()[0]["id"]
        term = "Affiliate"
        bm25_tf = self.keyword_search.get_bm25_tf(file_id, term)
        self.assertIsInstance(bm25_tf, float)

    def test_bm25(self):
        file_id = load_parsed_pdfs()[0]["id"]
        term = "Affiliate"
        bm25 = self.keyword_search.bm25(file_id, term)
        self.assertIsInstance(bm25, float)

    def test_bm25_search(self):
        query = "Affiliate definition"
        search_results = self.keyword_search.bm25_search(query)
        self.assertIsInstance(search_results, list)

if __name__ == "__main__":
    unittest.main()