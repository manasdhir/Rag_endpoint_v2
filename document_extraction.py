# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.pipeline_options import (
#     PdfPipelineOptions, 
#     LayoutOptions, 
#     TableStructureOptions, 
#     TableFormerMode, 
#     EasyOcrOptions
# )
from langchain_core.documents import Document
# from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
# from docling.chunking import HybridChunker
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import VlmPipelineOptions
# from docling.pipeline.vlm_pipeline import VlmPipeline
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter  # New import for the splitter
from doc_anal import DocumentIntelligenceService
from chonkie import SDPMChunker
from config import azure_open_ai_key,azure_open_ai_url
client=DocumentIntelligenceService()
# from patch_azure import AzureAIEmbeddings
# embeddings=AzureAIEmbeddings(
#     model="text-embedding-3-large",
#     base_url=azure_open_ai_url,
#     api_key=azure_open_ai_key,
#     dimension=768
#)
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
import hashlib
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)
URI = "./bajaj_embed_512_new"
conn=lancedb.connect(URI)
from langchain_community.vectorstores import LanceDB
# embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

from chonkie import SentenceTransformerEmbeddings
# embeddings = SentenceTransformerEmbeddings(
#     "BAAI/bge-m3",
#     device="cuda:0"
# )
# chunker = SDPMChunker(
#     embedding_model="BAAI/bge-m3",  # Default model
#     threshold=0.5,                              # Similarity threshold (0-1)
#     chunk_size=512,                             # Maximum tokens per chunk
#     min_sentences=5,                            # Initial sentences per chunk
#     skip_window=1                               # Number of chunks to skip when looking for similarities
# )
from threading import Lock
background_task_lock = Lock()
# ocr_options = EasyOcrOptions(force_full_page_ocr=True)
# accelerator_options = AcceleratorOptions(
#         num_threads=8, device=AcceleratorDevice.CUDA
#     )
# pipeline_options = PdfPipelineOptions()
# pipeline_options.do_ocr = True  # Enables OCR (now forced on all pages via options)
# pipeline_options.ocr_options = ocr_options
# pipeline_options.do_table_structure = True
# pipeline_options.table_structure_options = TableStructureOptions(
#     mode=TableFormerMode.ACCURATE,  # Precise table reconstruction
#     do_cell_matching=True           # Preserves cell details accurately
# )
# # Instantiate the converter with VlmPipeline and the pipeline options
# converter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(
#             pipeline_options=pipeline_options  # Apply the configured options
#         )
#     }
# )

# def convert_to_dl_doc(url: str):
#     doc = converter.convert(url)
#     return doc.document

def convert_to_md(url):
    analysis=client.analyze(source=url)
    content=analysis["analyzeResult"]["content"]
    return content

# def chunk_them(markdown_output):
#     # Export to Markdown to preserve formatting (tables, headers, etc.)
    
#     # Set up the splitter with prioritized separators and size/overlap
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2000,  # Approx. 2000 words (characters-based)
#         chunk_overlap=500,  # Approx. 200 words for context
#         separators=[
#             "\n# ",     # Prioritize top-level headers first
#             "\n## ",    # Then second-level
#             "\n### ",   # Third-level
#             "\n#### ",  # Fourth-level
#             "\n##### ", # Fifth-level (deepest)
#             "\n\n",     # Then paragraphs
#             "\n",       # Lines
#             ".",        # Sentences
#             " ",        # Words
#             ""          # Fallback to characters
#         ],
#         keep_separator=True  # Retains headers and Markdown in chunks
#     )
    
#     # Split into formatted chunks
#     formatted_chunks = splitter.split_text(markdown_output)
#     documents = [Document(page_content=chunk, metadata={"source": "Document"}) for chunk in formatted_chunks]
#     return documents
def chunk_them(doc):
    chunks=chunker.chunk(doc)
    text_list = [chunk.text for chunk in chunks]
    return text_list

def extract_to_markdown(url, text):
    with background_task_lock:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc = convert_to_md(url)
        chunks = chunk_them(doc)
        table_name = f"table_{text_hash[:8]}"
        db = LanceDB.from_texts(
            chunks,
            embedding=embeddings_model,
            connection=conn,
            table_name=table_name
        )
        table = db._table
        table.create_fts_index("text", with_position=True)

        redis_client.set(text_hash, table_name)
        print("document cached")
        return

def extract_to_markdown_local(text):
    # with background_task_lock:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(text_hash)
        table_name = f"UNITED_insurance"
        redis_client.set(text_hash, table_name)
        print("document cached")
        return

def extract_to_markdown_local_2(text):
    with background_task_lock:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with open('newton.txt') as f:
            doc=f.read()
        chunks = chunk_them(doc)
        table_name = f"table_{text_hash[:8]}"
        db = LanceDB.from_texts(
            chunks,
            embedding=embeddings_model,
            connection=conn,
            table_name=table_name
        )
        table = db._table
        table.create_fts_index("text", with_position=True)

        redis_client.set(text_hash, table_name)
        print("document cached")
        return  
def delete_document_cache(text_hash: str):
    """
    Deletes a cached document using its text_hash.
    Removes both the Redis mapping and the LanceDB table.
    """
    # 1️⃣ Get table name from redis
    table_name = redis_client.get(text_hash)
    if table_name:
        table_name = table_name.decode("utf-8")
        # 3️⃣ Delete the key from Redis
        redis_client.delete(text_hash)
        print(f"Deleted Redis entry for {text_hash}")
    else:
        print(f"No cache found for text_hash: {text_hash}")

if __name__ == "__main__":
    # import fitz
    # def extract_text_from_pdf(pdf_path):
    #     text = ""
    #     with fitz.open(pdf_path) as doc:
    #         for page in doc:
    #             text += page.get_text()
    #     return text
    # text=extract_text_from_pdf('files/UNI GROUP HEALTH INSURANCE POLICY - UIIHLGP26043V022526 1.pdf')
    # extract_to_markdown_local(text)
    # # delete_document_cache("3d5ba84af4a248e3b9a126b4ca2cd4fdfbdd6d7309bee2f559c1540d09806c20")
    # # redis_client.flushdb()
    # def print_all_redis_entries():
    #     print("=== Redis Entries ===")
    #     for key in redis_client.scan_iter("*"):  # Scan all keys
    #         try:
    #             value = redis_client.get(key)
    #             print(f"{key.decode('utf-8')} => {value.decode('utf-8') if value else None}")
    #         except Exception as e:
    #             print(f"{key} => <non-string value or error: {e}>")
    print_all_redis_entries()
    # from pprint import pprint
    # import time
    # t1=time.time()
    # doc = extract_to_markdown(url="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D")
    # t2=time.time()
    # print("time taken",t2-t1)
    # pprint(doc)  # Added closing parenthesis for correctness
    #print(extract_to_markdown_local())