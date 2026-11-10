"""Build the three indexes (knowledge, case documents, past cases) and retrieve passages."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import db
from .case_types import load_case_types
from .settings import Settings

logger = logging.getLogger(__name__)

INDEXES = ("knowledge", "case_docs", "past_cases")
NO_MIN_SCORE = -1.0  # pass as min_score to keep the top passages whatever their score
_SECTION_ID = re.compile(r"\b([A-Z]{2,4}-[A-Z]?\d{1,3})\b")


@dataclass
class Passage:
    """One retrieved chunk of a document."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float


_models: dict[str, Any] = {}
_clients: dict[str, Any] = {}
_clients_lock = threading.RLock()
_build_lock = threading.Lock()


# ----- models and storage -----


def get_embedder(settings: Settings) -> Any:
    """Load the embedding model once."""
    key = f"embed:{settings.embedding_model}"
    if key not in _models:
        from huggingface_hub.constants import HF_HUB_CACHE
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Without cache_folder, LlamaIndex downloads to its own folder outside the allowed caches.
        _models[key] = HuggingFaceEmbedding(
            model_name=settings.embedding_model, device="cpu", embed_batch_size=16, cache_folder=HF_HUB_CACHE
        )
    return _models[key]


def get_reranker(settings: Settings, top_n: int) -> Any:
    """Load the reranker once, and set how many passages it returns."""
    key = f"rerank:{settings.reranker_model}"
    if key not in _models:
        from llama_index.core.postprocessor import SentenceTransformerRerank

        _models[key] = SentenceTransformerRerank(model=settings.reranker_model, top_n=top_n, device="cpu")
    reranker = _models[key]
    reranker.top_n = top_n
    return reranker


def _client(settings: Settings) -> Any:
    key = str(settings.index_dir)
    with _clients_lock:
        if key not in _clients:
            import chromadb

            settings.index_dir.mkdir(parents=True, exist_ok=True)
            _clients[key] = chromadb.PersistentClient(
                path=key, settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
        return _clients[key]


def _collection(index: str, settings: Settings) -> Any:
    return _client(settings).get_or_create_collection(index, metadata={"hnsw:space": "cosine"})


def _store(index: str, settings: Settings) -> Any:
    from llama_index.vector_stores.chroma import ChromaVectorStore

    return ChromaVectorStore(chroma_collection=_collection(index, settings))


def _query_store(index: str, query: Any, settings: Settings) -> Any:
    """Query an index, reopening the store once if it has lost its in-memory search index.

    Seen in a long-running process: the query fails with "Error creating hnsw segment
    reader: Nothing found on disk", yet a new client reads the same folder correctly. The
    cause inside the vector store is not known, so only this exact error is retried.
    """
    from chromadb.errors import InternalError

    client = _client(settings)
    try:
        return _store(index, settings).query(query)
    except InternalError as exc:
        if "hnsw segment reader" not in str(exc):
            raise
        logger.warning("Reopening the vector store after: %s", exc)
    with _clients_lock:
        key = str(settings.index_dir)
        if _clients.get(key) is client:  # another thread may already have replaced it
            del _clients[key]
            client.close()
    return _store(index, settings).query(query)


# ----- turning files into chunks -----


def _splitter(settings: Settings) -> Any:
    from llama_index.core.node_parser import SentenceSplitter

    return SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()


def read_text_file(path: Path) -> str:
    """Read a .md, .txt or .pdf file as text."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def _sections(text: str) -> list[str]:
    """Split text at Markdown headings, keeping each heading with its body."""
    parts = re.split(r"(?m)^(?=#{1,4} )", text)
    return [part.strip() for part in parts if part.strip()]


def knowledge_chunks(path: Path, case_type: str, source: str, settings: Settings) -> list[dict[str, Any]]:
    """Chunk a policy wording or a warning-sign guide by heading."""
    splitter = _splitter(settings)
    chunks: list[dict[str, Any]] = []
    slug = _slug(path.stem)
    for section in _sections(read_text_file(path)):
        heading = section.split("\n", 1)[0]
        if "\n" not in section and (section.startswith("#") or len(section) < 120):
            continue  # a heading with no body, or the one-line sample notice, carries no content
        found = _SECTION_ID.search(heading) if heading.startswith("#") else None
        for piece in splitter.split_text(section):
            metadata = {"case_type": case_type, "source": source, "file": path.name, "path": str(path)}
            if found:
                metadata["clause_id" if source == "policy" else "flag_id"] = found.group(1)
            chunks.append({
                "id": f"knowledge:{case_type}:{source}:{slug}:{len(chunks)}",
                "text": piece, "metadata": metadata,
            })
    return chunks


def case_chunks(path: Path, case_id: str, case_type: str, settings: Settings) -> list[dict[str, Any]]:
    """Chunk one case document."""
    doc = path.stem
    return [
        {
            "id": f"case_docs:{case_id}:{doc}:{number}", "text": piece,
            "metadata": {"case_id": case_id, "case_type": case_type, "doc_type": doc, "path": str(path)},
        }
        for number, piece in enumerate(_splitter(settings).split_text(read_text_file(path)))
    ]


def past_case_chunk(record: dict[str, Any]) -> dict[str, Any]:
    """One chunk per past case."""
    past_id = record["past_case_id"]
    return {
        "id": f"past_cases:{past_id}:0",
        "text": f"{record['summary']} Outcome: {record['outcome']}. {record.get('note', '')}".strip(),
        "metadata": {
            "past_case_id": past_id, "case_type": record["case_type"], "outcome": record["outcome"],
            "provider": record.get("provider", ""),
        },
    }


# ----- finding what needs indexing -----


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _knowledge_files(settings: Settings) -> list[tuple[Path, str, str]]:
    files: list[tuple[Path, str, str]] = []
    folders: list[tuple[Path, str, str]] = [(settings.knowledge_dir / "general" / "red_flags", "general", "red_flags")]
    for name, case_type in load_case_types(settings).items():
        base = settings.root / case_type.knowledge_folder
        folders += [(base / "policy", name, "policy"), (base / "red_flags", name, "red_flags")]
    for folder, case_type_name, source in folders:
        if folder.exists():
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in {".md", ".txt", ".pdf"}:
                    files.append((path, case_type_name, source))
    return files


def _sources(settings: Settings) -> list[tuple[str, str, str, Any]]:
    """Every source item as (key, index, content hash, function that makes its chunks)."""
    items: list[tuple[str, str, str, Any]] = []
    for path, case_type, source in _knowledge_files(settings):
        key = f"knowledge:{case_type}:{source}:{path.name}"
        items.append((key, "knowledge", _hash(path.read_bytes()),
                      lambda p=path, c=case_type, s=source: knowledge_chunks(p, c, s, settings)))
    for record_path in sorted(settings.cases_dir.glob("*/case.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for path in sorted(record_path.parent.glob("*.md")):
            key = f"case_docs:{record['case_id']}:{path.stem}"
            items.append((key, "case_docs", _hash(path.read_bytes()),
                          lambda p=path, r=record: case_chunks(p, r["case_id"], r["case_type"], settings)))
    if settings.past_cases_path.exists():
        for record in db.read_jsonl(settings.past_cases_path):
            key = f"past_cases:{record['past_case_id']}"
            items.append((key, "past_cases", _hash(json.dumps(record, sort_keys=True).encode("utf-8")),
                          lambda r=record: [past_case_chunk(r)]))
    return items


def _remove_source(key: str, index: str, conn: Any, settings: Settings) -> None:
    rows = conn.execute("SELECT id FROM passages WHERE metadata_json LIKE ?", (f'%"source_key": "{key}"%',)).fetchall()
    ids = [row["id"] for row in rows]
    if ids:
        _collection(index, settings).delete(ids=ids)
        conn.executemany("DELETE FROM passages WHERE id = ?", [(i,) for i in ids])
    conn.execute("DELETE FROM indexed_files WHERE key = ?", (key,))


def build_indexes(settings: Settings, force: bool = False) -> str:
    """Add new or changed sources to the indexes. Unchanged files are skipped."""
    with _build_lock:  # the web app can ask for this from more than one request at once
        return _build_indexes(settings, force)


def _build_indexes(settings: Settings, force: bool) -> str:
    from llama_index.core.schema import TextNode

    db.sync_cases(settings)
    db.sync_past_cases(settings)
    conn = db.connect(settings)
    try:
        known = {row["key"]: row["content_hash"] for row in conn.execute("SELECT key, content_hash FROM indexed_files")}
        items = _sources(settings)
        current = {item[0] for item in items}
        indexes_of = {row["key"]: row["index_name"] for row in conn.execute("SELECT key, index_name FROM indexed_files")}
        with conn:
            for key in set(known) - current:
                _remove_source(key, indexes_of[key], conn, settings)
        todo = [item for item in items if force or known.get(item[0]) != item[2]]
        counts = {name: 0 for name in INDEXES}
        embedder = get_embedder(settings) if todo else None
        for start in range(0, len(todo), 32):
            for key, index, content_hash, make_chunks in todo[start : start + 32]:
                with conn:
                    _remove_source(key, index, conn, settings)
                    chunks = make_chunks()
                    if chunks:
                        vectors = embedder.get_text_embedding_batch([c["text"] for c in chunks])
                        nodes = []
                        for chunk, vector in zip(chunks, vectors):
                            metadata = {**chunk["metadata"], "source_key": key}
                            nodes.append(TextNode(id_=chunk["id"], text=chunk["text"], metadata=metadata, embedding=vector))
                            conn.execute(
                                "INSERT OR REPLACE INTO passages (id, index_name, text, metadata_json) VALUES (?, ?, ?, ?)",
                                (chunk["id"], index, chunk["text"], json.dumps(metadata)),
                            )
                        _store(index, settings).add(nodes)
                    conn.execute(
                        "INSERT OR REPLACE INTO indexed_files (key, index_name, content_hash) VALUES (?, ?, ?)",
                        (key, index, content_hash),
                    )
                counts[index] += 1
        skipped = len(items) - len(todo)
    finally:
        conn.close()
    return f"Indexed {counts}; skipped {skipped} unchanged sources."


def indexes_built(settings: Settings) -> bool:
    """True if every index has at least one passage."""
    if not settings.db_path.exists():
        return False
    rows = db.query(settings, "SELECT index_name, COUNT(*) AS n FROM passages GROUP BY index_name")
    return {row["index_name"] for row in rows if row["n"]} >= set(INDEXES)


# ----- retrieval -----


def _to_filters(filters: dict[str, Any] | None) -> Any:
    if not filters:
        return None
    from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

    items = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            items.append(MetadataFilter(key=key, value=list(value), operator=FilterOperator.IN))
        else:
            items.append(MetadataFilter(key=key, value=value, operator=FilterOperator.EQ))
    return MetadataFilters(filters=items, condition=FilterCondition.AND)


def _rerank_score(raw: float, temperature: float) -> float:
    """Map the reranker's raw relevance value to a 0 to 1 score.

    The raw value is an unbounded logit. Dividing by a temperature spreads the scale so
    that min_score keeps answerable questions and stops only clearly unrelated ones.
    """
    return 1.0 / (1.0 + math.exp(-raw / temperature))


def _cosine_score(similarity: float) -> float:
    """The vector store reports exp(-distance); turn that back into cosine similarity."""
    return 1.0 + math.log(max(similarity, 1e-9))


def retrieve(
    index: str, question: str, filters: dict[str, Any] | None = None, *, settings: Settings,
    top_n: int | None = None, use_reranker: bool | None = None, min_score: float | None = None,
) -> list[Passage]:
    """Embed the question, take top_k by similarity, rerank, and drop low scores."""
    from llama_index.core.schema import NodeWithScore
    from llama_index.core.vector_stores import VectorStoreQuery

    collection = _collection(index, settings)
    if collection.count() == 0:
        return []
    top_n = top_n or settings.top_n_after_rerank
    embedding = get_embedder(settings).get_query_embedding(question)
    result = _query_store(
        index,
        VectorStoreQuery(query_embedding=embedding, similarity_top_k=settings.top_k, filters=_to_filters(filters)),
        settings,
    )
    nodes = [
        NodeWithScore(node=node, score=score)
        for node, score in zip(result.nodes or [], result.similarities or [])
    ]
    rerank = settings.use_reranker if use_reranker is None else use_reranker
    if rerank and nodes:
        nodes = get_reranker(settings, top_n).postprocess_nodes(nodes, query_str=question)
        scored = [(n, _rerank_score(float(n.score), settings.rerank_temperature)) for n in nodes]
    else:
        scored = [(n, _cosine_score(float(n.score))) for n in nodes[:top_n]]
    threshold = settings.min_score if min_score is None else min_score
    return [
        Passage(id=n.node.node_id, text=n.node.get_content(), metadata=dict(n.node.metadata), score=score)
        for n, score in scored
        if score >= threshold
    ]


def best_match(text: str, passages: list[Passage], settings: Settings) -> tuple[Passage, float]:
    """The passage the reranker scores highest against a text, with its 0 to 1 score."""
    from llama_index.core.schema import NodeWithScore, TextNode

    nodes = [NodeWithScore(node=TextNode(id_=p.id, text=p.text), score=p.score) for p in passages]
    top = get_reranker(settings, 1).postprocess_nodes(nodes, query_str=text)[0]
    chosen = next(p for p in passages if p.id == top.node.node_id)
    return chosen, _rerank_score(float(top.score), settings.rerank_temperature)


def retrieve_many(
    index: str, questions: Iterable[str], filters: dict[str, Any] | None, settings: Settings, limit: int = 8,
) -> list[Passage]:
    """Run several searches and return the best distinct passages."""
    best: dict[str, Passage] = {}
    for question in questions:
        for passage in retrieve(index, question, filters, settings=settings, min_score=NO_MIN_SCORE):
            if passage.id not in best or passage.score > best[passage.id].score:
                best[passage.id] = passage
    return sorted(best.values(), key=lambda p: p.score, reverse=True)[:limit]


def get_passage(passage_id: str, settings: Settings) -> dict[str, Any] | None:
    """Look up a passage by its stable ID."""
    rows = db.query(settings, "SELECT id, index_name, text, metadata_json FROM passages WHERE id = ?", (passage_id,))
    if not rows:
        return None
    row = rows[0]
    row["metadata"] = json.loads(row.pop("metadata_json"))
    return row
