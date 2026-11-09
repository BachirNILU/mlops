"""05 - Vector stores: ChromaDB, pgvector, Azure AI Search behind one interface.

WHAT IT SHOWS
  The same chunks and vectors loaded into each store, the same query, the same top-3.
  Each store has the same two methods: add(chunks, vectors) and query(vector, k).
  - numpy:            brute force in memory (baseline)
  - ChromaDB:         embedded vector DB in a local folder (runs/chroma). Always runs.
  - pgvector:         PostgreSQL extension. Runs if PGVECTOR_URL is set (docker compose up -d db).
  - Azure AI Search:  managed cloud service. Runs if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY are set.

RUN
  python scripts/05_vector_stores.py
  (optional) docker compose up -d db   then set PGVECTOR_URL in .env and run again

EXPECTED OUTPUT
  Identical top-3 for numpy and ChromaDB; pgvector and Azure print "skipped" unless configured.

WHAT IT DOES NOT SHOW
  Scale (millions of vectors), index tuning (HNSW parameters), metadata filters, access control
  per document, or updates and deletes. Azure AI Search code is untested here (needs a subscription).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.config import banner  # noqa: E402
from lab.rag import (  # noqa: E402
    AzureSearchStore,
    ChromaStore,
    NumpyStore,
    PgVectorStore,
    build_chunks,
    chunk_text_for_embedding,
    get_embedder,
)

QUERY = "What should I check first when the guide bearing temperature rises slowly?"


def main() -> None:
    banner("05 - Vector stores")
    chunks = build_chunks()
    emb = get_embedder().fit([chunk_text_for_embedding(c) for c in chunks])
    vectors = emb.embed([chunk_text_for_embedding(c) for c in chunks])
    qv = emb.embed([QUERY])[0]
    dim = vectors.shape[1]
    print(f"{len(chunks)} chunks, {dim} dimensions. Query: {QUERY!r}\n")

    stores = [("numpy", lambda: NumpyStore()), ("chroma", lambda: ChromaStore())]
    if os.getenv("PGVECTOR_URL"):
        stores.append(("pgvector", lambda: PgVectorStore(os.environ["PGVECTOR_URL"], dim)))
    else:
        print("pgvector: skipped (set PGVECTOR_URL; see docker-compose.yml)")
    if os.getenv("AZURE_SEARCH_ENDPOINT") and os.getenv("AZURE_SEARCH_KEY"):
        stores.append(
            (
                "azure",
                lambda: AzureSearchStore(
                    os.environ["AZURE_SEARCH_ENDPOINT"], os.environ["AZURE_SEARCH_KEY"], dim
                ),
            )
        )
    else:
        print(
            "Azure AI Search: skipped (set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY; costs money)"
        )

    for label, make in stores:
        try:
            store = make()
            store.add(chunks, vectors)
            hits = store.query(qv, k=3)
            print(f"\n{store.name}:")
            for c, s in hits:
                print(f"   {s:.3f}  {c.chunk_id:<14} {c.text[:60]!r}")
        except Exception as e:
            print(f"\n{label}: failed: {type(e).__name__}: {e}")
    print(
        "\nTake-away: the store is a swappable part. Choose by operations (where data lives, scale,"
    )
    print(
        "security, hybrid search built in), not by retrieval quality, which comes from chunks and embeddings."
    )


if __name__ == "__main__":
    main()
