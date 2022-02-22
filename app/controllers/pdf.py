import pdfplumber
from .preprocessing import remove_special_characters

def parse_pdf(path: str):
    with pdfplumber.open(path) as pdf:
        first_page = pdf.pages[0]
        text = remove_special_characters(first_page.extract_text())
        return text