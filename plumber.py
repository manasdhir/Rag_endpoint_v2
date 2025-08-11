import requests
import os
import pdfplumber
import fitz
def download_pdf(url, save_path="downloaded.pdf"):
    """Download PDF from URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

def pdf_to_markdown_or_string(url):
    pdf_path=download_pdf(url)
    """Extract text from PDF and save to markdown or return as string"""
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    
        return all_text

def test(url):
    """Extract text from PDF and save to markdown or return as string"""
    all_text = ""
    with pdfplumber.open(url) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    
        return all_text
def get_pdf_page_count(pdf_path):
    with fitz.open(pdf_path) as doc:
        return len(doc)
if __name__ == "__main__":
    # Example PDF link
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    print(get_pdf_page_count("Test Case HackRx.pdf"))
    # Step 3: Or get as string
    # extracted_text = pdf_to_markdown_or_string(pdf_path)
    # print(extracted_text)
