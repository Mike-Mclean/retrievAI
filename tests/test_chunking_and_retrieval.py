from cli.lib.pdf_semantic_search import *
from cli.lib.search_utils import load_parsed_pdfs

import unittest

class TestSemanticSearch(unittest.TestCase):

    def setup(self):
        self.semantic_search = PdfSemanticSearch()
        self.documents = load_parsed_pdfs()

    def test_build_chunk_embeddings(self):
        embeddings = self.semantic_search.build_chunk_embeddings(self.documents)
        print(embeddings[0])

    def test_load_or_create_embeddings(self):
        embeddings = self.semantic_search.load_or_create_chunk_embeddings(self.documents)
        print(embeddings[0])

    def test_search_chunks(self):
        query = "affiliate"
        search_results = self.semantic_search.search_chunks(query)
        for result in search_results:
            print(result)

if __name__ == "__main__":
    unittest.main()
