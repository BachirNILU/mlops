"""RAG building blocks: chunking, embeddings, vector stores, BM25, hybrid search, reranking.

Read this file alongside scripts 04, 05 and 06.

Models used for retrieval are chosen by RAG_MODELS (independent of LAB_MODE / --real):
  RAG_MODELS=mock  (default) LSA embeddings (TF-IDF + truncated SVD, a pre-neural method) and a
                   rule-based reranker. Offline, instant, much weaker on synonyms.
  RAG_MODELS=local neural embedding model BAAI/bge-small-en-v1.5 and cross-encoder reranker
                   Xenova/ms-marco-MiniLM-L-6-v2, run locally with fastembed.
                   Free, no API key, but downloads ~150 MB from Hugging Face the first time.
                   Needs: pip install -r requirements-optional.txt
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lab.config import DOCS, RUNS

# ---------------------------------------------------------------- chunking


@dataclass
class Chunk:
    chunk_id: str  # e.g. "MAN-T-01#2"  (document ID + position)
    doc_id: str
    title: str  # document title, kept with every chunk for context
    section: str  # nearest heading
    text: str
    meta: dict = field(default_factory=dict)


def load_documents(folder: Path = DOCS) -> dict[str, str]:
    if not folder.exists():
        raise SystemExit("No documents found. Run: python scripts/00_make_data.py")
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))}


def chunk_fixed(doc_id: str, text: str, size: int = 300, overlap: int = 50) -> list[Chunk]:
    """Naive chunking: cut every `size` characters. Fast, but cuts sentences and lists."""
    title = text.splitlines()[0].lstrip("# ").strip()
    chunks, start, i = [], 0, 0
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(Chunk(f"{doc_id}#{i}", doc_id, title, "", piece.strip()))
        start += size - overlap
        i += 1
    return chunks


def chunk_by_heading(doc_id: str, text: str, max_chars: int = 700) -> list[Chunk]:
    """Structure-aware chunking: one chunk per '##' section, split further only if too long.
    Each chunk keeps its document title and section heading, so it makes sense on its own."""
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()
    sections: list[tuple[str, list[str]]] = [("intro", [])]
    for line in lines[1:]:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for section, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        paragraphs = [body[i : i + max_chars] for i in range(0, len(body), max_chars)]
        for part in paragraphs:
            cid = f"{doc_id}#{len(chunks)}"
            chunks.append(Chunk(cid, doc_id, title, section, part))
    return chunks


def build_chunks(strategy: str = "heading") -> list[Chunk]:
    docs = load_documents()
    out: list[Chunk] = []
    for doc_id, text in docs.items():
        out += (
            chunk_by_heading(doc_id, text) if strategy == "heading" else chunk_fixed(doc_id, text)
        )
    return out


def chunk_text_for_embedding(c: Chunk) -> str:
    """What we actually embed: title + section + text (adds context to short chunks)."""
    return f"{c.title}. {c.section}. {c.text}"


# ---------------------------------------------------------------- embeddings

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class MockEmbedder:
    """LSA embeddings: TF-IDF matrix -> SVD -> dense vectors. Must be fitted on the corpus."""

    name = "mock-lsa-64"

    def __init__(self, dim: int = 64):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=0)
        self.fitted = False

    def fit(self, texts: list[str]) -> MockEmbedder:
        x = self.vec.fit_transform(texts)
        self.svd.n_components = min(self.svd.n_components, x.shape[1] - 1, x.shape[0] - 1)
        self.svd.fit(x)
        self.fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        assert self.fitted, "call fit() on the corpus first"
        v = self.svd.transform(self.vec.transform(texts))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


class LocalNeuralEmbedder:
    """Real neural embeddings, computed locally (no API cost)."""

    name = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(self.name)

    def fit(self, texts: list[str]) -> LocalNeuralEmbedder:
        return self  # pretrained: nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        v = np.array(list(self.model.embed(texts)))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def local_models() -> bool:
    return os.getenv("RAG_MODELS", "mock").lower() == "local"


def _load(cls, *args):
    try:
        return cls(*args)
    except ImportError as e:
        raise SystemExit(
            "RAG_MODELS=local needs fastembed: pip install -r requirements-optional.txt"
        ) from e
    except ValueError as e:  # fastembed raises ValueError when the download fails
        raise SystemExit(
            f"Could not download the model ({e}). Check access to huggingface.co, "
            "or use RAG_MODELS=mock."
        ) from e


def get_embedder():
    return _load(LocalNeuralEmbedder) if local_models() else MockEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- vector stores
# Same three methods for every backend, so the rest of the code does not care which one runs.


class NumpyStore:
    """Brute-force search in memory. The baseline every store must beat or match."""

    name = "numpy (in memory)"

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks += chunks
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def query(self, vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ vector
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class ChromaStore:
    """ChromaDB: an embedded vector database. Stores vectors + text + metadata on disk."""

    name = "ChromaDB (local file)"

    def __init__(self, collection: str = "maintenance_docs", persist: bool = True):
        import chromadb

        self.client = (
            chromadb.PersistentClient(path=str(RUNS / "chroma"))
            if persist
            else chromadb.EphemeralClient()
        )
        try:
            self.client.delete_collection(collection)  # rebuild from scratch each run
        except Exception:
            pass
        # cosine distance; we pass our own vectors so Chroma does not download a model
        self.col = self.client.create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.by_id: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "section": c.section} for c in chunks],
        )
        self.by_id.update({c.chunk_id: c for c in chunks})

    def query(self, vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        res = self.col.query(query_embeddings=[vector.tolist()], n_results=k)
        return [
            (self.by_id[i], 1.0 - d)
            for i, d in zip(res["ids"][0], res["distances"][0], strict=False)
        ]


class PgVectorStore:
    """pgvector: vector search inside PostgreSQL. Good when your data already lives in Postgres.
    Needs a running database: `docker compose up -d db` and PGVECTOR_URL in .env."""

    name = "pgvector (PostgreSQL)"

    def __init__(self, url: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute("DROP TABLE IF EXISTS chunks")
        self.conn.execute(
            f"CREATE TABLE chunks (id text PRIMARY KEY, doc_id text, section text,"
            f" body text, embedding vector({dim}))"
        )
        self.by_id: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            for c, v in zip(chunks, vectors, strict=False):
                cur.execute(
                    "INSERT INTO chunks VALUES (%s, %s, %s, %s, %s)",
                    (c.chunk_id, c.doc_id, c.section, c.text, v.astype(np.float32)),
                )
        self.by_id.update({c.chunk_id: c for c in chunks})

    def query(self, vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        # <=> is pgvector's cosine distance operator
        rows = self.conn.execute(
            "SELECT id, 1 - (embedding <=> %s) AS score FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (vector.astype(np.float32), vector.astype(np.float32), k),
        ).fetchall()
        return [(self.by_id[r[0]], float(r[1])) for r in rows]


class AzureSearchStore:
    """Azure AI Search: managed search service with vector + keyword (hybrid) search built in.
    Needs AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY in .env. Not runnable offline.
    NOTE: written against azure-search-documents 12.0.0 but not run in this lab's CI (needs Azure)."""

    name = "Azure AI Search (cloud)"

    def __init__(self, endpoint: str, key: str, dim: int, index: str = "maintenance-docs"):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        cred = AzureKeyCredential(key)
        idx_client = SearchIndexClient(endpoint, cred)
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="body", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dim,
                vector_search_profile_name="default",
            ),
        ]
        vs = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
        )
        idx_client.create_or_update_index(SearchIndex(name=index, fields=fields, vector_search=vs))
        self.client = SearchClient(endpoint, index, cred)
        self.by_id: dict[str, Chunk] = {}

    @staticmethod
    def _key(chunk_id: str) -> str:
        return chunk_id.replace("#", "_")  # '#' is not allowed in Azure document keys

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.client.upload_documents(
            [
                {
                    "id": self._key(c.chunk_id),
                    "doc_id": c.doc_id,
                    "body": c.text,
                    "embedding": v.tolist(),
                }
                for c, v in zip(chunks, vectors, strict=False)
            ]
        )
        self.by_id.update({self._key(c.chunk_id): c for c in chunks})

    def query(
        self, vector: np.ndarray, k: int = 5, text: str | None = None
    ) -> list[tuple[Chunk, float]]:
        from azure.search.documents.models import VectorizedQuery

        vq = VectorizedQuery(vector=vector.tolist(), k_nearest_neighbors=k, fields="embedding")
        # passing search_text as well makes Azure run hybrid search (BM25 + vector, fused with RRF)
        res = self.client.search(search_text=text, vector_queries=[vq], top=k)
        return [(self.by_id[r["id"]], float(r["@search.score"])) for r in res]


# ---------------------------------------------------------------- keyword search (BM25)


class BM25Index:
    """Keyword search. Finds exact terms (asset IDs, codes, numbers) that embeddings can blur."""

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk_text_for_embedding(c)) for c in chunks])

    def query(self, text: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(text))
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------- hybrid + rerank


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[Chunk, float]], k: int = 60
) -> list[tuple[Chunk, float]]:
    """RRF: score = sum over lists of 1 / (k + rank). Uses ranks, so score scales do not matter."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.chunk_id] = chunk
    order = sorted(scores, key=scores.get, reverse=True)
    return [(by_id[i], scores[i]) for i in order]


class MockReranker:
    """Stand-in for a cross-encoder: scores how well each passage covers the query terms,
    with extra weight on rare terms and on exact phrases. A real cross-encoder reads query and
    passage together with a neural network."""

    name = "mock-term-coverage"

    def __init__(self, corpus: list[Chunk]):
        n = len(corpus)
        df: dict[str, int] = {}
        for c in corpus:
            for t in set(tokenize(chunk_text_for_embedding(c))):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + n / d) for t, d in df.items()}

    def score(self, query: str, chunk: Chunk) -> float:
        q = [t for t in tokenize(query) if len(t) > 2]
        text = chunk_text_for_embedding(chunk).lower()
        toks = set(tokenize(text))
        cover = sum(self.idf.get(t, 1.0) for t in q if t in toks)
        total = sum(self.idf.get(t, 1.0) for t in q) or 1.0
        bigrams = [f"{a} {b}" for a, b in zip(q, q[1:], strict=False)]
        phrase_bonus = sum(0.15 for bg in bigrams if bg in text)
        return cover / total + phrase_bonus

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], k: int = 5):
        scored = [(c, self.score(query, c)) for c, _ in candidates]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


class LocalCrossEncoderReranker:
    """Real cross-encoder reranker, run locally with fastembed (downloads about 80 MB once)."""

    name = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, corpus: list[Chunk] | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(self.name)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], k: int = 5):
        texts = [chunk_text_for_embedding(c) for c, _ in candidates]
        scores = list(self.model.rerank(query, texts))
        scored = [(c, float(s)) for (c, _), s in zip(candidates, scores, strict=False)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def get_reranker(corpus: list[Chunk]):
    return _load(LocalCrossEncoderReranker, corpus) if local_models() else MockReranker(corpus)


# ---------------------------------------------------------------- the full retriever


class Retriever:
    """Everything together: chunks -> embeddings -> store + BM25 -> hybrid (RRF) -> rerank."""

    def __init__(self, store_factory=None):
        self.chunks = build_chunks("heading")
        self.embedder = get_embedder().fit([chunk_text_for_embedding(c) for c in self.chunks])
        vectors = self.embedder.embed([chunk_text_for_embedding(c) for c in self.chunks])
        self.store = (
            store_factory(vectors.shape[1]) if store_factory else ChromaStore(persist=False)
        )
        self.store.add(self.chunks, vectors)
        self.bm25 = BM25Index(self.chunks)
        self.reranker = get_reranker(self.chunks)

    def vector_search(self, query: str, k: int = 5):
        return self.store.query(self.embedder.embed([query])[0], k)

    def keyword_search(self, query: str, k: int = 5):
        return self.bm25.query(query, k)

    def hybrid_search(self, query: str, k: int = 5, pool: int = 20):
        fused = reciprocal_rank_fusion(
            self.vector_search(query, pool), self.keyword_search(query, pool)
        )
        return fused[:k]

    def search(self, query: str, k: int = 4, pool: int = 20):
        """Production path: hybrid search for recall, then rerank for precision."""
        candidates = self.hybrid_search(query, k=pool, pool=pool)
        return self.reranker.rerank(query, candidates, k)


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
