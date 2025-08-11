import time
import datetime
import logging
from fastapi import FastAPI, Header, HTTPException, status, Request
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
from LLM import generate_search_queries,construct_prompts,generate_answers
import concurrent.futures
from config import azure_open_ai_key, azure_open_ai_url
# embeddings = AzureOpenAIEmbeddings(
#     model="text-embedding-3-large",
#     azure_endpoint=azure_open_ai_url,
#     api_key=azure_open_ai_key,
#     dimensions=768,
#     openai_api_version="2023-05-15"
# )
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
URI = "./bajaj_embed"
conn=lancedb.connect(URI)
table=""
reranker = ColbertReranker()
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    table_name = f"temp_table_{uuid.uuid4().hex}"
    md_chunks=extract_to_markdown(document_link)
    print("extraction done")
    db=LanceDB.from_texts(md_chunks,embedding=embeddings,connection=conn, table_name=table_name)
    global table
    table=db._table
    table.create_fts_index("text", with_position=True)
    print("index done")
    queries=generate_search_queries(questions)
    def perform_search(single_query):
        # Hybrid search for one query
        query=embeddings.embed_query(single_query)
        results = table.search(query_type="hybrid").vector(query).text(single_query).limit(5).rerank(reranker).to_pandas()["text"].to_list()
        return results
    all_results=[perform_search(i) for i in queries]
    print("search done")
    prompts=construct_prompts(questions=questions,contexts=all_results)
    answers=generate_answers(prompts=prompts)
    conn.drop_table(table_name)
    sent_at = datetime.datetime.utcnow()
    end=time.time()
    print("latency",end-start)
    print("sent_at",sent_at)
    print(answers)
    return {"answers": answers}
    


    