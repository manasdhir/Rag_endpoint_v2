import fitz
import hashlib

def hash_pdf_content(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

file1 = "Test Case HackRx.pdf"
file2 = "principia_newton.pdf"

print(hash_pdf_content(file1) == hash_pdf_content(file2))