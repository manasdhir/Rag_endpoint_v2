import asyncio
import time
import datetime
import json
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from config import REQUIRED_BEARER_TOKEN, azure_open_ai_key, azure_open_ai_url
from document_extraction import extract_to_markdown
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb
from lancedb.rerankers import ColbertReranker
from LLM import generate_search_queries, construct_prompts, generate_answers

# --- Logging to terminal setup ---
import logging
logger = logging.getLogger("hackrx")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# Embedding & vector setup
# embeddings = AzureOpenAIEmbeddings(
#     model="text-embedding-3-large",
#     azure_endpoint=azure_open_ai_url,
#     api_key=azure_open_ai_key,
#     dimensions=768,
#     openai_api_version="2023-05-15"
# )
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
URI = "./bajaj_embed"
conn = lancedb.connect(URI)
table = ""
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

# File logging (JSONL)
LOG_PATH = Path("hackrx_stream_log.jsonl")
_file_lock = asyncio.Lock()

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

async def append_log(entry: dict):
    async with _file_lock:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        def write():
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        await asyncio.to_thread(write)

@app.post("/api/v1/hackrx/run")
async def run_hackrx(
    request: Request,
    payload: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    received_at_dt = datetime.datetime.utcnow()
    received_at = received_at_dt.isoformat() + "Z"
    start = time.time()

    # immediate terminal log
    logger.info(f"Request received: document={payload.documents} questions={payload.questions}")

    verify_bearer_token(authorization)

    document_link = payload.documents
    questions = payload.questions

    async def streamer():
        # initial heartbeat
        yield b" "
        heartbeat_interval = 25  # keep less than gateway idle 60s
        last_hb = asyncio.get_event_loop().time()

        # Run the core pipeline in a thread to avoid blocking event loop
        def pipeline_work():
            logger.info("Starting extraction")
            md_chunks = extract_to_markdown(document_link)
            logger.info("Extraction done")
            table_name = f"temp_table_{uuid.uuid4().hex}"
            db = LanceDB.from_texts(md_chunks, embedding=embeddings, connection=conn, table_name=table_name)
            global table
            table = db._table
            logger.info("Creating FTS index")
            table.create_fts_index("text", with_position=True)
            logger.info("Index done")
            # generate queries
            queries = generate_search_queries(questions)
            logger.info(f"Generated search queries: {queries}")
            all_results = []
            for q in queries:
                logger.info(f"Performing hybrid search for query: {q}")
                query_vec = embeddings.embed_query(q)
                results = (
                    table.search(query_type="hybrid")
                    .vector(query_vec)
                    .text(q)
                    .limit(5)
                    .rerank(reranker)
                    .to_pandas()["text"]
                    .to_list()
                )
                all_results.append(results)
            logger.info("Search done")
            prompts = construct_prompts(questions=questions, contexts=all_results)
            logger.info("Constructed prompts")
            answers = generate_answers(prompts=prompts)
            logger.info("Generated answers")
            # cleanup
            conn.drop_table(table_name)
            logger.info("Dropped temp table")
            return queries, all_results, answers

        pipeline_task = asyncio.create_task(asyncio.to_thread(pipeline_work))

        # keep-alive loop
        while not pipeline_task.done():
            now = asyncio.get_event_loop().time()
            if now - last_hb >= heartbeat_interval:
                yield b" "
                last_hb = now
            await asyncio.sleep(0.5)

        try:
            queries, all_results, answers = await pipeline_task
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            error_payload = {"error": str(e)}
            yield (json.dumps(error_payload) + "\n").encode()
            return

        sent_at_dt = datetime.datetime.utcnow()
        sent_at = sent_at_dt.isoformat() + "Z"
        end = time.time()
        duration_s = end - start

        # Log to terminal summary
        logger.info(f"Pipeline complete: duration_s={duration_s:.2f} answers={answers}")

        # File log entry
        log_entry = {
            "received_at": received_at,
            "document_link": document_link,
            "questions": questions,
            "search_queries": queries,
            "answers": answers,
            "response_ready_at": sent_at,
            "duration_s": duration_s,
        }
        await append_log(log_entry)

        # Final response (strict format)
        yield json.dumps({"answers": answers}, ensure_ascii=False).encode()

    return StreamingResponse(streamer(), media_type="application/json")
