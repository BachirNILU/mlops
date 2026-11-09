"""04 - Chunking and embeddings.

WHAT IT SHOWS
  1. Two ways to split a document: fixed size (cuts mid-sentence) vs by heading (keeps sections).
  2. Embeddings: each chunk becomes a vector; similar meaning -> high cosine similarity.
  3. For three questions, the most similar chunks.

RUN
  python scripts/04_chunking_embeddings.py                    (LSA embeddings, offline)
  RAG_MODELS=local python scripts/04_chunking_embeddings.py   (local neural model, free, ~70 MB download)

EXPECTED OUTPUT
  Chunk lists for MAN-T-01 (fixed vs heading), vector shape, and a top-3 list per question.
  The cooling-water question should rank PROC-CW-04 or MAN-T-01 near the top.

WHAT IT DOES NOT SHOW
  PDF/table extraction, choosing chunk size by evaluation (script 06 does a simple version),
  or embedding API costs. LSA (mock) is far weaker than neural embeddings on synonyms.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from lab.config import banner  # noqa: E402
from lab.rag import (  # noqa: E402
    build_chunks,
    chunk_by_heading,
    chunk_fixed,
    chunk_text_for_embedding,
    get_embedder,
    load_documents,
)


def main() -> None:
    banner("04 - Chunking and embeddings")
    text = load_documents()["MAN-T-01"]
    print("Fixed-size chunks (300 chars, 50 overlap) of MAN-T-01:")
    for c in chunk_fixed("MAN-T-01", text)[:4]:
        print(f"  {c.chunk_id:<11} ...{c.text[-60:]!r}")
    print("\nHeading-based chunks of MAN-T-01:")
    for c in chunk_by_heading("MAN-T-01", text):
        print(f"  {c.chunk_id:<11} section={c.section!r:<38} {len(c.text)} chars")

    chunks = build_chunks("heading")
    texts = [chunk_text_for_embedding(c) for c in chunks]
    emb = get_embedder().fit(texts)
    vectors = emb.embed(texts)
    print(
        f"\nEmbedder: {emb.name}. {len(chunks)} chunks -> matrix {vectors.shape} (one row per chunk)"
    )

    for q in [
        "cooling water flow problem makes bearing hot",
        "when must the turbine be stopped because of vibration",
        "is it allowed to open a bearing without paperwork",
    ]:
        qv = emb.embed([q])[0]
        sims = vectors @ qv  # vectors are normalised, so dot product = cosine similarity
        top = np.argsort(-sims)[:3]
        print(f"\nQ: {q}")
        for i in top:
            print(f"   {sims[i]:.2f}  {chunks[i].chunk_id:<14} {chunks[i].text[:70]!r}")
    print(
        "\nTake-away: chunk on structure, keep the title with each chunk, and test retrieval on real questions."
    )


if __name__ == "__main__":
    main()
