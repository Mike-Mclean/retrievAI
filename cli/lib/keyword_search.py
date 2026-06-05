from collections import defaultdict, Counter
import string
from nltk.stem import PorterStemmer

from search_utils import load_stopwords


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict()

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
