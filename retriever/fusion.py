"""
Rank fusion. Deliberately free of AWS or OpenSearch imports so it can be
unit tested without credentials, network, or service dependencies.
"""
from __future__ import annotations

from typing import Any

FINAL_TOP_K = 5
RRF_K = 60


def hits_to_ranked_chunks(response: dict) -> list[dict[str, Any]]:
    """Normalise an OpenSearch response into chunks carrying their rank."""
    ranked = []
    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        src = hit["_source"]
        ranked.append({
            "id": hit["_id"],
            "text": src.get("text", ""),
            "source": src.get("source", "unknown"),
            "chunk_index": src.get("chunk_index", 0),
            "page": src.get("page"),
            "score": hit["_score"],
            "rank": rank,
        })
    return ranked


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[dict[str, Any]]],
    top_k: int = FINAL_TOP_K,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse ranked lists by rank rather than score.

    Each arm contributes 1 / (k + rank) per document. Scores from kNN
    (cosine similarity) and BM25 (unbounded, corpus-dependent) are not
    comparable, so ranks are the only sound basis for combining them.
    """
    fused: dict[str, dict[str, Any]] = {}

    for arm, chunks in ranked_lists.items():
        for chunk in chunks:
            entry = fused.get(chunk["id"])
            if entry is None:
                entry = {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "page": chunk["page"],
                    "rrf_score": 0.0,
                    "ranks": {},
                }
                fused[chunk["id"]] = entry

            entry["rrf_score"] += 1.0 / (k + chunk["rank"])
            entry["ranks"][arm] = chunk["rank"]

    ordered = sorted(
        fused.values(),
        key=lambda c: (-c["rrf_score"], c["source"], c["chunk_index"]),
    )
    return ordered[:top_k]