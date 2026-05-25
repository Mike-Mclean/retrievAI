from search_utils import list_pdf_documents, PDF_TESTING_PATH, CACHE_PATH
import os
import pdfplumber
from pathlib import Path
import json

def restructure_table(table: list) -> str:
    new_table = []
    headers = table[0]
    for row in table[1:]:
        row_str = ""
        for column, item in enumerate(row):
            row_str += f"{headers[column]}: {item} | "
        new_table.append(row_str)
    return new_table

def parse_pdf(file_path: str | Path | os.PathLike) -> list[dict]:
    page_data = []
    file = os.path.basename(file_path)
    with pdfplumber.open(file_path) as pdf:
        pages = pdf.pages
        for page in pages:
            tables = page.extract_tables() #Maybe crop pages first
            text = page.extract_text()

            restructured_tables = []
            for table in tables:
                restruct_table = restructure_table(table)
                restructured_tables.append(restruct_table)

            page_data.append({
                "file_name": file,
                "page": page.page_number,
                "text": text,
                "tables": restructured_tables
            })
    return page_data

def parse_all_pdfs(documents_path: str | Path | os.PathLike = PDF_TESTING_PATH,
                   cache_path: str | Path | os.PathLike = CACHE_PATH):
    docs = list_pdf_documents(documents_path)

    all_pdf_data = []

    for doc in docs:
        doc_path = os.path.join(documents_path, doc)
        doc_data = parse_pdf(doc_path)
        for data in doc_data:
            all_pdf_data.append(data)

    with open(cache_path, 'w') as cache:
        json.dump(all_pdf_data, cache, indent=2)

    return all_pdf_data

if __name__ == '__main__':
    parse_all_pdfs()