import pdfplumber
import re
from pathlib import Path

PDF_DIR = Path("data/reports")
TEXT_DIR = Path("data/text")
TEXT_DIR.mkdir(exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'page \d+|\d+/\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bcid\s*\d+\b', '', text)

    return text.strip()

def pdf_to_text(pdf_path):
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
    return clean_text(" ".join(full_text))

for pdf_file in PDF_DIR.glob("*.pdf"):
    text = pdf_to_text(pdf_file)
    out_file = TEXT_DIR / f"{pdf_file.stem}_clean.txt"
    out_file.write_text(text, encoding="utf-8")
    print(f"Saved: {out_file}")
