import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "retriever"))

from fusion import reciprocal_rank_fusion  # noqa: E402


def _chunk(id_, rank, source="doc.pdf", idx=0):
    return {
        "id": id_, "text": f"text-{id_}", "source": source,
        "chunk_index": idx, "page": 1, "rank": rank,
    }


def test_chunk_found_by_both_arms_ranks_first():
    knn = [_chunk("a", 1, idx=0), _chunk("b", 2, idx=1)]
    bm25 = [_chunk("c", 1, source="other.pdf", idx=5), _chunk("a", 2, idx=0)]

    fused = reciprocal_rank_fusion({"knn": knn, "bm25": bm25}, top_k=3)

    assert fused[0]["ranks"] == {"knn": 1, "bm25": 2}
    assert len(fused) == 3


def test_lexical_only_hit_reaches_results():
    """The whole point of hybrid: BM25 can surface what kNN never returned."""
    knn = [_chunk("a", 1, idx=0), _chunk("b", 2, idx=1)]
    bm25 = [_chunk("c", 1, source="other.pdf", idx=5)]

    fused = reciprocal_rank_fusion({"knn": knn, "bm25": bm25}, top_k=5)
    ids = [(c["source"], c["chunk_index"]) for c in fused]

    assert ("other.pdf", 5) in ids


def test_respects_top_k():
    knn = [_chunk(str(i), i, idx=i) for i in range(1, 11)]
    assert len(reciprocal_rank_fusion({"knn": knn}, top_k=3)) == 3