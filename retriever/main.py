import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from aws_xray_sdk.core import xray_recorder, patch_all

xray_recorder.configure(context_missing="LOG_ERROR")
patch_all()

from agent import retrieve  # noqa: E402 — must come after patch_all

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agent Retriever", version="1.0.0")


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
async def health_check():
    """ALB health check — must return HTTP 200."""
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    """
    Hybrid retrieval: embed the query, run kNN and BM25 as separate searches,
    fuse both ranked lists with RRF, return the top-N chunks.

    Deliberately not `async`: retrieve() is fully synchronous (boto3,
    opensearch-py), so as a coroutine it would block the event loop and
    serialise concurrent requests. A plain `def` lets FastAPI run it in a
    threadpool.
    """
    return retrieve(query=request.query, top_k=request.top_k)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
