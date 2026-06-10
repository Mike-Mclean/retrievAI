from cli.lib.keyword_search import *
from cli.lib.search_utils import load_parsed_pdfs, INDEX_PATH

import unittest

class TestInvertedIndex(unittest.TestCase):

    def setUp(self):
        self.keyword_search = InvertedIndex()
        self.documents = load_parsed_pdfs()

    def test_building_inverted_index(self):
        self.keyword_search.build()
        self.assertTrue(INDEX_PATH.is_file())

    def test_save_inverted_index(self):
        self.keyword_search.save()
        self.assertTrue(INDEX_PATH.is_file())

    def test_load_inverted_index(self):
        self.keyword_search.load()
        self.assertTrue(INDEX_PATH.is_file())

    def test_get_tf(self):
        file_name = "AudibleInc.pdf"
        term = "Affiliate"
        tf = self.keyword_search.get_tf(file_name, term)
        self.assertIsInstance(tf, int)

    def test_get_idf(self):
        file_name = "AudibleInc.pdf"
        term = "Affiliate"
        idf = self.keyword_search.get_idf(file_name, term)
        self.assertIsInstance(idf, float)

    def test_get_tf_idf(self):
        file_name = "AudibleInc.pdf"
        term = "Affiliate"
        tf_idf = self.keyword_search.get_tf_idf(file_name, term)
        self.assertIsInstance(tf_idf, float)

    def test_get_bm25_idf(self):
        term = "Affiliate"
        bm_25_idf = self.keyword_search.get_bm25_idf(term)
        self.assertIsInstance(bm_25_idf, float)


if __name__ == "__main__":
    unittest.main()