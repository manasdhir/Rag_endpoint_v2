import subprocess
import uuid
import os
import pathlib

def convert_to_pdf(document_path: str) -> str:
    """
    Converts a PPT, DOCX, XLSX, etc. to PDF using LibreOffice.
    Returns the unique PDF filename (not full path).
    """
    if not os.path.isfile(document_path):
        raise FileNotFoundError(f"{document_path} does not exist")

    # Generate a unique filename
    unique_name = f"{uuid.uuid4().hex}.pdf"
    output_dir = pathlib.Path(document_path).parent

    # Run LibreOffice to convert to PDF
    subprocess.run([
        'libreoffice',
        '--headless',
        '--convert-to', 'pdf',
        '--outdir', str(output_dir),
        document_path
    ], check=True)

    # Find the converted file (LibreOffice retains original base name)
    original_name = pathlib.Path(document_path).stem + ".pdf"
    original_pdf_path = output_dir / original_name

    return original_pdf_path

if __name__=="__main__":
    a=(convert_to_pdf('Mediclaim Insurance Policy.docx'))
    print(a)
