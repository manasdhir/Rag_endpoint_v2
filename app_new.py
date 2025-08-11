import time
import datetime
import logging
from fastapi import FastAPI, Header, HTTPException, status, Request,BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from config import REQUIRED_BEARER_TOKEN
from document_extraction import extract_to_markdown
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb
from lancedb.rerankers import ColbertReranker
import uuid
from LLM import construct_prompts, client_14,create_queries_parallel,generate_answers
import concurrent.futures
from config import azure_open_ai_key, azure_open_ai_url
from LLM import batch_embed_nested
import concurrent.futures
from threading import Lock
import concurrent.futures
from threading import Lock
from search import search_per_group_parallel
import re
import uuid
import os
import requests
import fitz
from langdetect import detect
from test_excel import convert_to_pdf
from gemini_clients import get_answers_from_pdf,get_answers_from_image_url,get_answers, get_query_generation, parse_response_text
import redis
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chonkie import SentenceChunker
from agent import create_graph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import hashlib
from document_extraction import conn

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=600
)

graph=create_graph()
from document_extraction import redis_client
def get_pdf_page_count(pdf_path):
    with fitz.open(pdf_path) as doc:
        return len(doc)
    

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def is_english(text: str, sample_size: int = 500) -> bool:
    sample = text[:sample_size]
    try:
        return detect(sample) == "en"
    except:
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_endpoint.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# embeddings = AzureOpenAIEmbeddings(
#     model="text-embedding-3-large",
#     azure_endpoint=azure_open_ai_url,
#     api_key=azure_open_ai_key,
#     dimensions=768,
#     openai_api_version="2023-05-15"
# )

table=""
#reranker = ColbertReranker()
app = FastAPI()

def get_table_name_by_text(text: str):
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    table_name = redis_client.get(text_hash)
    table_name=table_name.decode('utf-8') if table_name else None
    return table_name,text_hash


embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 256}
)
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sent_chunker = SentenceChunker(
    tokenizer_or_token_counter="character",  # Default tokenizer (or use "gpt2", etc.)
    chunk_size=512,                  # Maximum tokens per chunk
    chunk_overlap=128,               # Overlap between chunks
    min_sentences_per_chunk=1        # Minimum sentences in each chunk
)

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

def verify_bearer_token(authorization: Optional[str]) -> None:
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        scheme, token = authorization.split()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if scheme.lower() != "bearer" or token != REQUIRED_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/api/v1/hackrx/run")
async def run_hackrx(
    request: Request,
    payload: HackRxRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None)
):
    received_at = datetime.datetime.utcnow()
    start = time.time()
    print("received",received_at)
    verify_bearer_token(authorization)
    document_link = payload.documents
    questions = payload.questions
    print(document_link)
    print(questions)
   
    logger.info(f"Document Link: {document_link}")
    logger.info(f"Questions: {questions}")
    
    extra_pt=""
    extra_pt_2=""
    if re.search(r"\.(xlsx|pptx|png|jpe?g)([?#&]|$)", document_link, re.IGNORECASE):
        ext = re.search(r"\.(xlsx|pptx|png|jpe?g)", document_link, re.IGNORECASE).group(1).lower()
        if ext == "xlsx" or ext == "pptx":
            response = requests.get(document_link, stream=True, timeout=10)
            response.raise_for_status()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            os.makedirs("app", exist_ok=True)
            save_path = os.path.join("app", unique_filename)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=262144):
                    f.write(chunk)
            processed_pdf=convert_to_pdf(save_path)
            return get_answers_from_pdf(processed_pdf,questions)
        if ext=="png" or ext == "jpg" or ext == "jpeg":
            return get_answers_from_image_url(document_link,questions)
    elif re.search(r"\.(docx|pdf)([?#&]|$)", document_link, re.IGNORECASE):
            ext = re.search(r"\.(docx|pdf)", document_link, re.IGNORECASE).group(1).lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            os.makedirs("app", exist_ok=True)
            save_path = os.path.join("app", unique_filename)

            response = requests.get(document_link, stream=True, timeout=10)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=262144):
                    f.write(chunk)
            if ext=="docx":
                processed_pdf=convert_to_pdf(save_path)
            else:
                processed_pdf=save_path
            pg_count=get_pdf_page_count(processed_pdf)
            print(pg_count)
            if pg_count<10:
                text=extract_text_from_pdf(processed_pdf)
                text=text + f"\nBased on the above context answer the following questions in the specified format\n:{questions}"
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}} 
                result =await graph.ainvoke(
                        {"messages": [HumanMessage(content=text)]},
                        config=config
                    )
                result=result["messages"][-1].content
                print(result)
                ans=parse_response_text(result)
                print(ans)
                return {"answers":ans}
            else:
                text=extract_text_from_pdf(processed_pdf)
                table_from_cache,text_hash=get_table_name_by_text(text)
                if not table_from_cache:
                    flag=False
                    background_tasks.add_task(extract_to_markdown, document_link,text)
                    table_name = f"table_name_{uuid.uuid4().hex[:8]}"
                    chunks=text_splitter.split_text(text)
                    db=LanceDB.from_texts(chunks,embedding=embeddings_model,connection=conn, table_name=table_name)
                    table=db._table
                    table.create_fts_index("text", with_position=True)
                    queries = create_queries_parallel(questions)
                    embed = batch_embed_nested(queries=queries, embeddings_model=embeddings_model)
                    all_results = search_per_group_parallel(queries, embed, table)
                    prompts=construct_prompts(questions=questions,contexts=all_results)
                    answers=generate_answers(prompts=prompts, text_hash=text_hash, flag=flag,questions=questions)
                    conn.drop_table(table_name)
                    return {"answers":answers}
                else:
                    flag=True
                    table=conn.open_table(table_from_cache)
                    queries = create_queries_parallel(questions)
                    embed = batch_embed_nested(queries=queries, embeddings_model=embeddings_model)
                    all_results = search_per_group_parallel(queries, embed, table)
                    prompts=construct_prompts(questions=questions,contexts=all_results)
                    answers=generate_answers(prompts=prompts, text_hash=text_hash, flag=flag,questions=questions)
                    return {"answers":answers}
    elif re.search(r"\.(bin|zip)(\?|$)", document_link):
        return {"answers":["user uploaded an unsupported file" for _ in questions]}
    else:
        text=f"the user has provided this link {document_link} and the questions {questions}"
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}} 
        result =await graph.ainvoke(
                {"messages": [HumanMessage(content=text)]},
                config=config
            )
        print(result["messages"])
        result=result["messages"][-1].content
        ans=parse_response_text(result)
        return {"answers":ans}   
    


    