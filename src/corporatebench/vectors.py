from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import os
import re
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from filelock import FileLock

from .kb import KnowledgeBase

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PAPER_EMBEDDING_MODEL = "text-embedding-3-small"
PAPER_EMBEDDING_DIMENSIONS = 1536

_MODEL_ID = re.compile(
    r"(?=.{1,200}\Z)[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
)
_REVISION = re.compile(r"[0-9a-f]{40}")
_MANIFEST_FIELDS = {
    "version",
    "company",
    "document_fingerprint",
    "embedding_backend",
    "model",
    "revision",
    "metric",
    "dimensions",
    "total_documents",
    "indexed_documents",
    "complete",
}


class EmbeddingBackend(Protocol):
    name: str
    revision: str | None

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class VectorProgress:
    indexed_documents: int
    total_documents: int


@dataclass(frozen=True)
class VectorIndexInfo:
    path: Path
    model: str
    revision: str | None
    dimensions: int
    indexed_documents: int
    total_documents: int
    complete: bool


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        revision: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if not isinstance(model, str) or _MODEL_ID.fullmatch(model) is None:
            raise ValueError("model must be a Hugging Face model ID")
        if revision is not None and (
            not isinstance(revision, str) or _REVISION.fullmatch(revision) is None
        ):
            raise ValueError("revision must be a 40-character commit hash")
        if isinstance(batch_size, bool) or not 1 <= batch_size <= 1_024:
            raise ValueError("batch_size must be between 1 and 1024")
        if device not in {None, "cpu", "mps", "cuda"}:
            raise ValueError("device must be 'cpu', 'mps', 'cuda', or None")

        self.name = model
        self.revision = (
            DEFAULT_EMBEDDING_REVISION
            if model == DEFAULT_EMBEDDING_MODEL and revision is None
            else revision
        )
        self.batch_size = batch_size
        self.device = device
        self.cache_key = f"sentence-transformers:{self.name}:{self.revision}"
        self._model = None
        self._lock = threading.RLock()

    @property
    def dimensions(self) -> int:
        dimensions = self._load_model().get_embedding_dimension()
        if not isinstance(dimensions, int) or not 1 <= dimensions <= 65_536:
            raise ValueError("Embedding model returned an invalid dimension")
        return dimensions

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=False)

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip() or len(text) > 1_000:
            raise ValueError("Invalid search query")
        return self._encode([text], query=True)[0]

    def _encode(self, texts: Sequence[str], *, query: bool) -> np.ndarray:
        if not texts or len(texts) > 1_024:
            raise ValueError("Invalid embedding batch")
        if any(not isinstance(text, str) or "\x00" in text for text in texts):
            raise ValueError("Embedding inputs must be valid strings")
        with self._lock:
            model = self._load_model()
            method = model.encode_query if query else model.encode_document
            vectors = method(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def _load_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.name,
                    revision=self.revision,
                    device=self.device,
                    trust_remote_code=False,
                )
        return self._model


class OpenAIEmbedder:
    """The embedding backend used by the published benchmark."""

    def __init__(
        self,
        model: str = PAPER_EMBEDDING_MODEL,
        *,
        client: object | None = None,
        api_key: str | None = None,
        cache_key: str | None = None,
    ) -> None:
        if model != PAPER_EMBEDDING_MODEL:
            raise ValueError(f"paper embedding model must be {PAPER_EMBEDDING_MODEL}")
        self.name = model
        self.revision = None
        self.cache_key = cache_key or f"openai:{model}:{PAPER_EMBEDDING_DIMENSIONS}"
        if api_key is not None and (
            not isinstance(api_key, str)
            or not 1 <= len(api_key) <= 10_000
            or any(character in api_key for character in "\x00\r\n")
        ):
            raise ValueError("Invalid API key")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "Install CorporateBench with the 'openai' extra"
                ) from error
            options: dict[str, object] = {"timeout": 60.0, "max_retries": 2}
            if api_key is not None:
                options["api_key"] = api_key
            client = OpenAI(**options)
        self._client = client

    @property
    def dimensions(self) -> int:
        return PAPER_EMBEDDING_DIMENSIONS

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts or len(texts) > 1_024:
            raise ValueError("Invalid embedding batch")
        if any(not isinstance(text, str) or "\x00" in text for text in texts):
            raise ValueError("Embedding inputs must be valid strings")
        return self._embed(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip() or len(text) > 1_000:
            raise ValueError("Invalid search query")
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(input=texts, model=self.name)
        data = sorted(response.data, key=lambda item: int(item.index))
        if len(data) != len(texts):
            raise ValueError("Embedding API returned an invalid result count")
        return np.asarray([item.embedding for item in data], dtype=np.float32)


class VectorIndex:
    VERSION = 3
    MAX_MANIFEST_BYTES = 100_000

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        company: str,
        embedder: EmbeddingBackend,
        cache_dir: str | os.PathLike[str] | None = None,
        batch_size: int = 32,
    ) -> None:
        if (
            inspect.getattr_static(embedder, "dimensions", None) is None
            or not callable(getattr(embedder, "embed_documents", None))
            or not callable(getattr(embedder, "embed_query", None))
        ):
            raise TypeError("embedder must implement EmbeddingBackend")
        if (
            not isinstance(embedder.name, str)
            or not 1 <= len(embedder.name) <= 200
            or any(character in embedder.name for character in "\x00\r\n")
        ):
            raise ValueError("embedder.name is invalid")
        if embedder.revision is not None and (
            not isinstance(embedder.revision, str)
            or not 1 <= len(embedder.revision) <= 200
            or any(character in embedder.revision for character in "\x00\r\n")
        ):
            raise ValueError("embedder.revision is invalid")
        cache_key = getattr(embedder, "cache_key", None)
        if cache_key is not None and (
            not isinstance(cache_key, str)
            or not 1 <= len(cache_key) <= 500
            or any(character in cache_key for character in "\x00\r\n")
        ):
            raise ValueError("embedder.cache_key is invalid")
        if isinstance(batch_size, bool) or not 1 <= batch_size <= 1_024:
            raise ValueError("batch_size must be between 1 and 1024")

        default_cache = Path.home() / ".cache" / "corporatebench" / "vectors"
        selected_cache = default_cache if cache_dir is None else cache_dir
        self.cache_dir = _prepare_cache_dir(selected_cache)
        self.kb = kb
        self.company = company
        self.embedder = embedder
        self._embedding_backend = cache_key or _python_backend_key(embedder)
        self.batch_size = batch_size
        self._index_dir: Path | None = None
        self._matrix_path: Path | None = None
        self._norms_path: Path | None = None

    def bootstrap(
        self,
        progress: Callable[[VectorProgress], None] | None = None,
    ) -> VectorIndexInfo:
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")
        dimensions = self.embedder.dimensions
        if isinstance(dimensions, bool) or not 1 <= dimensions <= 65_536:
            raise ValueError("embedder.dimensions is invalid")
        expected = self._expected_manifest(dimensions)
        index_dir = self._resolve_index_dir(expected)
        index_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
        if index_dir.is_symlink() or not index_dir.resolve(strict=True).is_relative_to(
            self.cache_dir
        ):
            raise ValueError("Vector cache directory is unsafe")

        lock_path = self.cache_dir / f".{index_dir.name}.lock"
        if lock_path.is_symlink():
            raise ValueError("Vector cache lock is unsafe")
        with FileLock(lock_path, timeout=3_600):
            info = self._bootstrap_locked(index_dir, expected, progress)
        self._index_dir = index_dir
        self._matrix_path = index_dir / "embeddings.npy"
        self._norms_path = index_dir / "squared_norms.npy"
        return info

    def search(self, query: str, *, limit: int = 10) -> str:
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 1_000
            or "\x00" in query
        ):
            raise ValueError("Invalid search query")
        if isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("Search limit must be between 1 and 50")
        if self._matrix_path is None:
            self.bootstrap()
        assert self._matrix_path is not None and self._norms_path is not None

        query_vector = _validate_vector(
            self.embedder.embed_query(query), self.embedder.dimensions
        )
        matrix = np.load(self._matrix_path, mmap_mode="r", allow_pickle=False)
        norms = np.load(self._norms_path, mmap_mode="r", allow_pickle=False)
        count = min(limit, self.kb.document_count)
        distances = np.asarray(
            norms - 2 * (matrix @ query_vector) + np.dot(query_vector, query_vector),
            dtype=np.float32,
        )
        if count == len(distances):
            candidates = np.arange(len(distances))
        else:
            candidates = np.argpartition(distances, count - 1)[:count]
        indices = sorted(
            (int(index) for index in candidates),
            key=lambda index: (float(distances[index]), index),
        )
        return self.kb.render_documents(indices)

    def _bootstrap_locked(
        self,
        index_dir: Path,
        expected: dict[str, object],
        progress: Callable[[VectorProgress], None] | None,
    ) -> VectorIndexInfo:
        manifest_path = index_dir / "manifest.json"
        matrix_path = index_dir / "embeddings.npy"
        norms_path = index_dir / "squared_norms.npy"
        if (
            manifest_path.is_symlink()
            or matrix_path.is_symlink()
            or norms_path.is_symlink()
        ):
            raise ValueError("Vector cache files may not be symlinks")

        if manifest_path.exists():
            manifest = _read_manifest(manifest_path)
            _validate_manifest(manifest, expected)
            matrix = _open_matrix(matrix_path, expected, mode="r+")
            norms = _open_norms(norms_path, expected, mode="r+")
        else:
            matrix = np.lib.format.open_memmap(
                matrix_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    int(expected["total_documents"]),
                    int(expected["dimensions"]),
                ),
            )
            norms = np.lib.format.open_memmap(
                norms_path,
                mode="w+",
                dtype=np.float32,
                shape=(int(expected["total_documents"]),),
            )
            manifest = {**expected, "indexed_documents": 0, "complete": False}
            _write_manifest(manifest_path, manifest)

        start = int(manifest["indexed_documents"])
        total = int(expected["total_documents"])
        dimensions = int(expected["dimensions"])
        for offset in range(start, total, self.batch_size):
            end = min(offset + self.batch_size, total)
            texts = [
                self.kb.document_for_embedding(index) for index in range(offset, end)
            ]
            vectors = _validate_matrix(
                self.embedder.embed_documents(texts),
                rows=end - offset,
                dimensions=dimensions,
            )
            matrix[offset:end] = vectors
            norms[offset:end] = np.einsum("ij,ij->i", vectors, vectors)
            matrix.flush()
            norms.flush()
            manifest["indexed_documents"] = end
            manifest["complete"] = end == total
            _write_manifest(manifest_path, manifest)
            if progress is not None:
                progress(VectorProgress(end, total))

        del matrix, norms
        return VectorIndexInfo(
            path=index_dir,
            model=str(expected["model"]),
            revision=self.embedder.revision,
            dimensions=dimensions,
            indexed_documents=total,
            total_documents=total,
            complete=True,
        )

    def _expected_manifest(self, dimensions: int) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "company": self.company,
            "document_fingerprint": self.kb.document_fingerprint,
            "embedding_backend": self._embedding_backend,
            "model": self.embedder.name,
            "revision": self.embedder.revision,
            "metric": "l2",
            "dimensions": dimensions,
            "total_documents": self.kb.document_count,
        }

    def _resolve_index_dir(self, expected: dict[str, object]) -> Path:
        encoded = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        cache_key = hashlib.sha256(encoded).hexdigest()[:24]
        path = self.cache_dir / f"{self.company}-{cache_key}"
        if not path.resolve(strict=False).is_relative_to(self.cache_dir):
            raise ValueError("Vector cache path is unsafe")
        return path


def _prepare_cache_dir(path: str | os.PathLike[str]) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("vector_cache_dir is invalid")
    candidate = Path(raw).expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("vector_cache_dir may not be a symlink")
    candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("vector_cache_dir must be a directory")
    return resolved


def _python_backend_key(embedder: object) -> str:
    implementation = type(embedder)
    digest = hashlib.sha256(
        f"{implementation.__module__}.{implementation.__qualname__}".encode()
    )
    for name in ("dimensions", "embed_documents", "embed_query"):
        member = inspect.getattr_static(implementation, name, None)
        if isinstance(member, property):
            member = member.fget
        member = getattr(member, "__func__", member)
        code = getattr(member, "__code__", None)
        digest.update(name.encode())
        payload = marshal.dumps(code) if code is not None else repr(member).encode()
        digest.update(payload)
    return f"python:{digest.hexdigest()}"


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size > VectorIndex.MAX_MANIFEST_BYTES:
        raise ValueError("Vector cache manifest is invalid")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Vector cache manifest is invalid") from error
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("Vector cache manifest is invalid")
    return payload


def _validate_manifest(
    manifest: dict[str, object], expected: dict[str, object]
) -> None:
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Vector cache does not match the benchmark data")
    indexed = manifest.get("indexed_documents")
    complete = manifest.get("complete")
    total = expected["total_documents"]
    if (
        isinstance(indexed, bool)
        or not isinstance(indexed, int)
        or not 0 <= indexed <= total
        or not isinstance(complete, bool)
        or complete != (indexed == total)
    ):
        raise ValueError("Vector cache manifest is invalid")


def _open_matrix(path: Path, expected: dict[str, object], *, mode: str) -> np.memmap:
    if not path.is_file():
        raise ValueError("Vector cache matrix is missing")
    try:
        matrix = np.load(path, mmap_mode=mode, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("Vector cache matrix is invalid") from error
    shape = (int(expected["total_documents"]), int(expected["dimensions"]))
    if matrix.shape != shape or matrix.dtype != np.dtype(np.float32):
        raise ValueError("Vector cache matrix is invalid")
    return matrix


def _open_norms(path: Path, expected: dict[str, object], *, mode: str) -> np.memmap:
    if not path.is_file():
        raise ValueError("Vector cache norms are missing")
    try:
        norms = np.load(path, mmap_mode=mode, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("Vector cache norms are invalid") from error
    if norms.shape != (int(expected["total_documents"]),) or norms.dtype != np.dtype(
        np.float32
    ):
        raise ValueError("Vector cache norms are invalid")
    return norms


def _validate_matrix(vectors: np.ndarray, *, rows: int, dimensions: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (rows, dimensions) or not np.isfinite(matrix).all():
        raise ValueError("Embedding backend returned invalid vectors")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("Embedding backend returned a zero vector")
    return matrix


def _validate_vector(vector: np.ndarray, dimensions: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (dimensions,) or not np.isfinite(array).all():
        raise ValueError("Embedding backend returned an invalid query vector")
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("Embedding backend returned a zero query vector")
    return array


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".manifest-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
