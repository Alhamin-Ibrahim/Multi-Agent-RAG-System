from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import boto3
from fusion import FINAL_TOP_K, hits_to_ranked_chunks, reciprocal_rank_fusion
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

logger = logging.getLogger(__name__)

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "")
INDEX_NAME = os.environ.get("OPENSEARCH_INDEX", "documents")
REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "eu-west-1"
)

CANDIDATE_K = 20    # candidates fetched per retrieval arm before fusion
SOURCE_FIELDS = ["text", "source", "chunk_index", "page"]


# Clients are cached so we pay for credential resolution and the TLS
# handshake once per container, not once per request.
@lru_cache(maxsize=1)
def _bedrock():
    return boto3.client("bedrock-runtime", region_name=REGION)


@lru_cache(maxsize=1)
def _opensearch() -> OpenSearch:
    if not OPENSEARCH_ENDPOINT:
        raise RuntimeError("OPENSEARCH_ENDPOINT is not set")

    auth = AWSV4SignerAuth(boto3.Session().get_credentials(), REGION, "aoss")
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", ""), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=10,
    )


def embed_query(query: str) -> list[float]:
    response = _bedrock().invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": query}),
    )
    return json.loads(response["body"].read())["embedding"]


def knn_search(client: OpenSearch, query_vector: list[float], k: int = CANDIDATE_K):
    """Dense arm: semantic similarity over Titan embeddings."""
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": query_vector, "k": k}}},
        "_source": SOURCE_FIELDS,
    }
    return hits_to_ranked_chunks(client.search(index=INDEX_NAME, body=body))


def bm25_search(client: OpenSearch, query: str, k: int = CANDIDATE_K):
    """
    Lexical arm: OpenSearch's own BM25 over the analysed `text` field.

    This is a separate query, not a re-rank of the kNN results, so exact-term
    matches the vector search missed can still reach the final set.
    """
    body = {
        "size": k,
        "query": {"match": {"text": {"query": query}}},
        "_source": SOURCE_FIELDS,
    }
    return hits_to_ranked_chunks(client.search(index=INDEX_NAME, body=body))


def retrieve(query: str, top_k: int = FINAL_TOP_K) -> dict[str, Any]:
    """Hybrid retrieval: dense kNN + lexical BM25, fused with RRF."""
    logger.info("Retrieving for query: '%s'", query[:80])
    client = _opensearch()

    knn_chunks = knn_search(client, embed_query(query))

    # Degrade to dense-only rather than failing the whole request.
    try:
        bm25_chunks = bm25_search(client, query)
    except Exception:
        logger.exception("Lexical arm failed; continuing with kNN results only")
        bm25_chunks = []

    logger.info("Candidates: %d kNN, %d BM25", len(knn_chunks), len(bm25_chunks))

    if not knn_chunks and not bm25_chunks:
        logger.warning("No chunks found for query")
        return {"chunks": [], "sources": [], "count": 0}

    fused = reciprocal_rank_fusion(
        {"knn": knn_chunks, "bm25": bm25_chunks}, top_k=top_k
    )
    sources = list(dict.fromkeys(c["source"] for c in fused))

    logger.info("Returning %d chunks after fusion", len(fused))
    return {"chunks": fused, "sources": sources, "count": len(fused)}