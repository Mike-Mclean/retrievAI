from search_utils import list_pdf_documents, PDF_TESTING_PATH
import os
import pdfplumber
from pdfplumber.page import Page
from pathlib import Path

def restructure_table(table: list) -> str:
    new_table = []
    headers = table[0]
    for row in table[1:]:
        row_str = ""
        for column, item in enumerate(row):
            row_str += f"{headers[column]: {item}} | "
        new_table.append(row_str)
    return new_table

def parse_pdf(file_path: str | Path | os.PathLike):
    page_data = []
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
                "page": page.page_number,
                "text": text,
                "tables": restructured_tables
            })
    return page_data


if __name__ == '__main__':
    parse_pdf()