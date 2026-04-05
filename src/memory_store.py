from __future__ import annotations

import re
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from config import logger

# ── models ──

_MEMORY_TYPES = ("user", "feedback", "project", "reference")


class MemoryEntry(BaseModel):
    slug: str
    type: str
    content: str
    created: datetime
    updated: datetime


class SearchResult(BaseModel):
    slug: str
    type: str
    snippet: str
    score: float


# ── helpers ──

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 50) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:max_len]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pack_embedding(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _unpack_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _temporal_decay(updated_iso: str, halflife_days: int) -> float:
    try:
        updated = datetime.fromisoformat(updated_iso.replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400
        return 0.3 + 0.7 * math.exp(-0.693 * days / max(halflife_days, 1))
    except Exception:
        return 1.0


# ── store ──

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    slug TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    embedding BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    slug, content, type,
    content='memories', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, slug, content, type)
    VALUES (new.rowid, new.slug, new.content, new.type);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, slug, content, type)
    VALUES ('delete', old.rowid, old.slug, old.content, old.type);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, slug, content, type)
    VALUES ('delete', old.rowid, old.slug, old.content, old.type);
    INSERT INTO memories_fts(rowid, slug, content, type)
    VALUES (new.rowid, new.slug, new.content, new.type);
END;
"""

_MEMORY_MD_HEADER = "# monoclaw memory\n\n"


class MemoryStore:
    def __init__(self, base_path: Path, halflife_days: int = 30,
                 embedding_weight: float = 0.6, mmr_lambda: float = 0.7) -> None:
        self._base = base_path
        self._base.mkdir(parents=True, exist_ok=True)
        self._db_path = base_path / "search.db"
        self._index_path = base_path / "MEMORY.md"
        self._halflife = halflife_days
        self._embedding_weight = embedding_weight
        self._mmr_lambda = mmr_lambda
        self._db = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    # ── CRUD ──

    def create(self, entry: MemoryEntry, embedding: np.ndarray | None = None) -> None:
        slug = self._unique_slug(entry.slug)
        entry = entry.model_copy(update={"slug": slug})
        self._write_md_file(entry)
        blob = _pack_embedding(embedding) if embedding is not None else None
        self._db.execute(
            "INSERT INTO memories (slug, type, content, created, updated, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entry.slug, entry.type, entry.content,
             entry.created.isoformat(), entry.updated.isoformat(), blob),
        )
        self._db.commit()

    def update(self, slug: str, content: str, embedding: np.ndarray | None = None) -> None:
        now = _now_iso()
        row = self._db.execute("SELECT type, created FROM memories WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            logger.warning(f"memory update: slug not found: {slug}")
            return
        entry = MemoryEntry(slug=slug, type=row[0], content=content,
                            created=datetime.fromisoformat(row[1]), updated=datetime.fromisoformat(now))
        self._write_md_file(entry)
        blob = _pack_embedding(embedding) if embedding is not None else None
        if blob is not None:
            self._db.execute(
                "UPDATE memories SET content=?, updated=?, embedding=? WHERE slug=?",
                (content, now, blob, slug),
            )
        else:
            self._db.execute(
                "UPDATE memories SET content=?, updated=? WHERE slug=?",
                (content, now, slug),
            )
        self._db.commit()

    def delete(self, slug: str) -> None:
        md_file = self._base / f"{slug}.md"
        if md_file.exists():
            md_file.unlink()
        self._db.execute("DELETE FROM memories WHERE slug = ?", (slug,))
        self._db.commit()

    def get(self, slug: str) -> MemoryEntry | None:
        row = self._db.execute(
            "SELECT slug, type, content, created, updated FROM memories WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return None
        return MemoryEntry(slug=row[0], type=row[1], content=row[2],
                           created=datetime.fromisoformat(row[3]), updated=datetime.fromisoformat(row[4]))

    def list_all(self) -> list[MemoryEntry]:
        rows = self._db.execute(
            "SELECT slug, type, content, created, updated FROM memories ORDER BY updated DESC"
        ).fetchall()
        return [MemoryEntry(slug=r[0], type=r[1], content=r[2],
                            created=datetime.fromisoformat(r[3]), updated=datetime.fromisoformat(r[4]))
                for r in rows]

    # ── search ──

    def search(self, query: str, query_embedding: np.ndarray | None = None,
               limit: int = 10) -> list[SearchResult]:
        # Step 1: FTS5 keyword candidates
        fts_results = self._fts_search(query, limit=limit * 3)

        # Step 2: vector similarity (if embeddings available)
        vector_scores: dict[str, float] = {}
        if query_embedding is not None:
            vector_scores = self._vector_scores(query_embedding)

        # Step 3: hybrid merge
        candidates = self._hybrid_merge(fts_results, vector_scores)

        # Step 4: MMR re-ranking
        if query_embedding is not None and candidates:
            candidates = self._mmr_rerank(candidates, limit)
        else:
            candidates = candidates[:limit]

        return candidates

    def _fts_search(self, query: str, limit: int) -> list[dict]:
        # tokenize into individual terms joined by OR for broad matching
        terms = [t.strip() for t in re.split(r'\s+', query.strip()) if t.strip()]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms)
        try:
            rows = self._db.execute(
                "SELECT m.slug, m.type, m.updated, "
                "  snippet(memories_fts, 1, '>>>', '<<<', '...', 40) as snippet, "
                "  bm25(memories_fts, 1.0, 3.0, 1.0) as rank "
                "FROM memories_fts "
                "JOIN memories m ON memories_fts.slug = m.slug "
                "WHERE memories_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        results = []
        for r in rows:
            decay = _temporal_decay(r[2], self._halflife)
            bm25_raw = abs(r[4])  # bm25 returns negative values (lower = better)
            results.append({
                "slug": r[0], "type": r[1],
                "snippet": r[3], "bm25": bm25_raw, "decay": decay,
            })
        # normalize bm25 scores to 0-1
        if results:
            max_bm25 = max(r["bm25"] for r in results) or 1.0
            for r in results:
                r["bm25_norm"] = r["bm25"] / max_bm25
        return results

    def _vector_scores(self, query_embedding: np.ndarray) -> dict[str, float]:
        rows = self._db.execute(
            "SELECT slug, embedding, updated FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        scores: dict[str, float] = {}
        for slug, blob, updated in rows:
            emb = _unpack_embedding(blob)
            sim = _cosine_similarity(query_embedding, emb)
            decay = _temporal_decay(updated, self._halflife)
            scores[slug] = max(0.0, sim) * decay
        return scores

    def _hybrid_merge(self, fts_results: list[dict],
                      vector_scores: dict[str, float]) -> list[SearchResult]:
        alpha = self._embedding_weight
        merged: dict[str, SearchResult] = {}

        for r in fts_results:
            slug = r["slug"]
            bm25_score = r.get("bm25_norm", 0.0) * r["decay"]
            vec_score = vector_scores.get(slug, 0.0)
            if vector_scores:
                score = (1 - alpha) * bm25_score + alpha * vec_score
            else:
                score = bm25_score
            merged[slug] = SearchResult(
                slug=slug, type=r["type"],
                snippet=r["snippet"], score=score,
            )

        # add vector-only results not found by FTS
        for slug, vec_score in vector_scores.items():
            if slug not in merged and vec_score > 0.3:
                entry = self.get(slug)
                if entry:
                    merged[slug] = SearchResult(
                        slug=slug, type=entry.type,
                        snippet=entry.content[:80] + "..." if len(entry.content) > 80 else entry.content,
                        score=alpha * vec_score,
                    )

        return sorted(merged.values(), key=lambda r: r.score, reverse=True)

    def _mmr_rerank(self, candidates: list[SearchResult], limit: int) -> list[SearchResult]:
        embeddings = self._load_embeddings([c.slug for c in candidates])
        if not embeddings:
            return candidates[:limit]

        lam = self._mmr_lambda
        selected: list[SearchResult] = []
        remaining = list(candidates)

        while remaining and len(selected) < limit:
            best_idx = 0
            best_mmr = -float("inf")
            for i, cand in enumerate(remaining):
                relevance = cand.score
                max_sim = 0.0
                cand_emb = embeddings.get(cand.slug)
                if cand_emb is not None:
                    for sel in selected:
                        sel_emb = embeddings.get(sel.slug)
                        if sel_emb is not None:
                            sim = _cosine_similarity(cand_emb, sel_emb)
                            max_sim = max(max_sim, sim)
                mmr = lam * relevance - (1 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    def _load_embeddings(self, slugs: list[str]) -> dict[str, np.ndarray]:
        if not slugs:
            return {}
        placeholders = ",".join("?" for _ in slugs)
        rows = self._db.execute(
            f"SELECT slug, embedding FROM memories WHERE slug IN ({placeholders}) AND embedding IS NOT NULL",
            slugs,
        ).fetchall()
        return {slug: _unpack_embedding(blob) for slug, blob in rows}

    # ── index generation ──

    def generate_index_md(self) -> str:
        entries = self.list_all()
        if not entries:
            return ""
        by_type: dict[str, list[MemoryEntry]] = {}
        for e in entries:
            by_type.setdefault(e.type, []).append(e)

        lines = []
        for t in _MEMORY_TYPES:
            group = by_type.get(t, [])
            lines.append(f"\n## {t}")
            if not group:
                lines.append("(none)")
            else:
                for e in group:
                    date = e.updated.strftime("%Y-%m-%d")
                    lines.append(f"- **{e.slug}** (updated {date})")
        logger.info(f"generated memory index: {', '.join(e.slug for e in entries)}")
        return f"{_MEMORY_MD_HEADER.rstrip()}" + "\n".join(lines) + "\n"

    def write_index_file(self) -> None:
        self._index_path.write_text(self.generate_index_md())

    def rebuild_index(self) -> None:
        """Re-sync SQLite from .md files on disk. Recovery/maintenance tool."""
        self._db.execute("DELETE FROM memories")
        self._db.commit()

        for md_file in self._base.glob("*.md"):
            if md_file.name == "MEMORY.md":
                continue
            try:
                entry = self._parse_md_file(md_file)
                if entry:
                    self._db.execute(
                        "INSERT INTO memories (slug, type, content, created, updated) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (entry.slug, entry.type, entry.content,
                         entry.created.isoformat(), entry.updated.isoformat()),
                    )
            except Exception as exc:
                logger.warning(f"rebuild_index: failed to parse {md_file.name}: {exc}")

        self._db.commit()
        self.write_index_file()
        logger.info("memory index rebuilt from disk")

    # ── file I/O helpers ──

    def _write_md_file(self, entry: MemoryEntry) -> None:
        frontmatter = {
            "type": entry.type,
            "created": entry.created.isoformat(),
            "updated": entry.updated.isoformat(),
        }
        text = "---\n" + yaml.dump(frontmatter, default_flow_style=False).rstrip() + "\n---\n\n" + entry.content + "\n"
        (self._base / f"{entry.slug}.md").write_text(text)

    def _parse_md_file(self, path: Path) -> MemoryEntry | None:
        text = path.read_text()
        if not text.startswith("---"):
            return None
        parts = text.split("---", 2)
        if len(parts) < 3:
            return None
        fm = yaml.safe_load(parts[1])
        if not isinstance(fm, dict):
            return None
        content = parts[2].strip()
        return MemoryEntry(
            slug=path.stem,
            type=fm.get("type", "user"),
            content=content,
            created=datetime.fromisoformat(str(fm.get("created", _now_iso()))),
            updated=datetime.fromisoformat(str(fm.get("updated", _now_iso()))),
        )

    def _unique_slug(self, desired: str) -> str:
        slug = _slugify(desired) or "memory"
        if not self._db.execute("SELECT 1 FROM memories WHERE slug=?", (slug,)).fetchone():
            return slug
        for i in range(2, 100):
            candidate = f"{slug}-{i}"
            if not self._db.execute("SELECT 1 FROM memories WHERE slug=?", (candidate,)).fetchone():
                return candidate
        return f"{slug}-{_now_iso().replace(':', '').replace('-', '')}"
