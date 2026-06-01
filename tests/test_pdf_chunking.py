from cli.lib.pdf_chunking import (
    semantic_chunk,
    normalize_pdf_text,
    detect_chunk_boundaries,
    chunk_by_paragraphs,
    split_into_chunks,
    merge_and_split,
    split_on_sentences
)

from cli.lib.search_utils import (
    load_parsed_pdfs
)

def test_pdf_chunking():
    pdf_data = load_parsed_pdfs()
    for i in range(13):
        text = pdf_data[i]['text']
        text = normalize_pdf_text(text)
        chunks = split_into_chunks(text)
        print(f"Page: {i + 1}")
        for chunk in chunks:
            print(chunk)
            print("\n\n")




if __name__ == '__main__':
    test_pdf_chunking()