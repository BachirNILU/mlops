"""06 - Hybrid search and reranking, measured.

WHAT IT SHOWS
  Four retrieval methods on the 18 answerable golden questions, scored at document level:
    vector only | BM25 keywords only | hybrid (Reciprocal Rank Fusion) | hybrid + rerank
  Metrics: recall@3 (share of relevant documents in the top 3) and MRR (1 / rank of first hit).
  Then one query where keywords matter (an exact code), side by side.

RUN
  python scripts/06_hybrid_rerank.py                    (LSA embeddings + rule-based reranker)
  RAG_MODELS=local python scripts/06_hybrid_rerank.py   (local neural embeddings + cross-encoder, free)

EXPECTED OUTPUT
  A 4-row table. Numbers differ between mock and real mode; read the table, do not assume
  hybrid always wins. With 18 questions, a difference of one question is 5 to 6 points.

WHAT IT DOES NOT SHOW
  Tuning (pool size, RRF k, chunk size) with a separate test set; that would avoid
  over-fitting to these 18 questions.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.config import EVAL, banner  # noqa: E402
from lab.metrics import dedupe, recall_at_k, reciprocal_rank  # noqa: E402
from lab.rag import Retriever  # noqa: E402

K = 3


def main() -> None:
    banner("06 - Hybrid search and reranking")
    gold = [json.loads(line) for line in (EVAL / "golden_set.jsonl").read_text().splitlines()]
    gold = [g for g in gold if g["relevant_docs"]]
    r = Retriever()
    methods = {
        "vector": lambda q: r.vector_search(q, 20),
        "bm25": lambda q: r.keyword_search(q, 20),
        "hybrid (RRF)": lambda q: r.hybrid_search(q, 20),
        "hybrid + rerank": lambda q: r.search(q, k=20),
    }
    print(f"{len(gold)} questions, embedder={r.embedder.name}, reranker={r.reranker.name}\n")
    print(f"{'method':<18}{'recall@' + str(K):>10}{'MRR':>8}")
    for name, fn in methods.items():
        recalls, rrs = [], []
        for g in gold:
            docs = dedupe([c.doc_id for c, _ in fn(g["question"])])
            recalls.append(recall_at_k(docs, set(g["relevant_docs"]), K))
            rrs.append(reciprocal_rank(docs, set(g["relevant_docs"])))
        print(f"{name:<18}{sum(recalls) / len(recalls):>10.2f}{sum(rrs) / len(rrs):>8.2f}")

    q = "P-101 vibration limit"
    print(f"\nExample query: {q!r}")
    for name in ["vector", "bm25", "hybrid + rerank"]:
        top = [c.chunk_id for c, _ in methods[name](q)[:3]]
        print(f"  {name:<16} {top}")
    print("\nTake-away: measure retrieval on your own questions before tuning the LLM part.")


if __name__ == "__main__":
    main()
