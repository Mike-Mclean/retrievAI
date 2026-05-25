import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PDF_TESTING_PATH = os.path.join(ROOT_PATH, "data", "pdf_testing")
CACHE_PATH = os.path.join(ROOT_PATH, "cache", "cache.json")

def list_pdf_documents(pdf_path):
    return os.listdir(pdf_path)