"""Disk cache for model outputs (scispaCy / de-id spans, SapBERT embeddings), shared by every run and study.

Keys include the model name and a hash of the detector module's source, so editing a detector invalidates its
entries. sqlite keeps concurrent runs safe. Disable with Config.cache = False (CLI --no-cache).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import pickle
import sqlite3
import threading
from functools import lru_cache

import numpy as np

from .config import CACHE_DIR

_lock = threading.Lock()


@lru_cache(maxsize=None)
def _db(name: str) -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_DIR / f"{name}.sqlite", timeout=60, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
    return con


def _get(db: str, key: str):
    row = _db(db).execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
    return None if row is None else row[0]


def _put_many(db: str, items: list[tuple[str, bytes]]):
    if not items:
        return
    with _lock:
        con = _db(db)
        con.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", items)
        con.commit()


@lru_cache(maxsize=None)
def _source_hash(fn) -> str:
    return hashlib.sha1(inspect.getsource(inspect.getmodule(fn)).encode()).hexdigest()[:12]


def spans(fn, text: str, field: str, *args, enabled: bool = True):
    """fn(text, field, *args) -> list[Span], cached on disk. Returns copies: callers mutate spans."""
    if not enabled:
        return fn(text, field, *args)
    key = hashlib.sha1(repr((fn.__module__, fn.__name__, _source_hash(fn), args, text)).encode()).hexdigest()
    blob = _get("spans", key)
    if blob is None:
        out = fn(text, field, *args)
        for s in out:
            s.extra.pop("match", None)
        blob = pickle.dumps([(s, s.field) for s in out])
        _put_many("spans", [(key, blob)])
    out = []
    for s, f in pickle.loads(blob):
        s = copy.deepcopy(s)
        s.field = field
        out.append(s)
    return out


class EmbeddingStore:
    """text -> unit vector, persisted per embedding model."""

    def __init__(self, model_name: str):
        self.prefix = model_name + "\x00"

    def get_many(self, texts: list[str]) -> dict[str, np.ndarray]:
        out = {}
        con = _db("embeddings")
        for i in range(0, len(texts), 500):
            chunk = texts[i:i + 500]
            q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(chunk))})"
            for k, v in con.execute(q, [self.prefix + t for t in chunk]):
                out[k[len(self.prefix):]] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]):
        _put_many("embeddings", [(self.prefix + t, v.astype(np.float32).tobytes()) for t, v in items.items()])
