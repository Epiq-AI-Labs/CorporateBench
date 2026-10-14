from __future__ import annotations

import gc
import hashlib
import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from pyoxigraph import RdfFormat, Store
from pyoxigraph import parse as parse_rdf

_ENTITY_PREFIX = "http://storybeat.company/entity/"
_QUERY_OPERATIONS = re.compile(
    r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE|INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|"
    r"COPY|MOVE|ADD|SERVICE|WITH)\b",
    re.IGNORECASE,
)
_FORBIDDEN_OPERATIONS = {
    "CONSTRUCT",
    "DESCRIBE",
    "INSERT",
    "DELETE",
    "LOAD",
    "CLEAR",
    "CREATE",
    "DROP",
    "COPY",
    "MOVE",
    "ADD",
    "SERVICE",
    "WITH",
}


class KnowledgeBase:
    """Read-only access to a StoryBeat .kb archive."""

    MAX_MEMBERS = 300_000
    MAX_UNCOMPRESSED_BYTES = 6_000_000_000
    MAX_MEMBER_BYTES = 3_500_000_000
    MAX_COMPRESSION_RATIO = 250
    MAX_DOCUMENT_BYTES = 2_000_000
    MAX_EMBEDDING_CONTENT_CHARS = 22_000
    MAX_QUERY_CHARS = 20_000
    MAX_RESULT_ROWS = 200
    MAX_RESULT_CHARS = 200_000
    MAX_SEARCH_RESULT_CHARS = 500_000

    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        if path.is_symlink() or not self.path.is_file() or self.path.suffix != ".kb":
            raise ValueError("KB must be a regular .kb file")
        if not zipfile.is_zipfile(self.path):
            raise ValueError("KB is not a valid ZIP archive")

        self._archive = zipfile.ZipFile(self.path)
        try:
            self._members = self._validate_archive()
        except Exception:
            self._archive.close()
            raise
        self._documents = tuple(
            sorted(name for name in self._members if name.startswith("documents/"))
        )
        with self.path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        self._fingerprint = fingerprint
        self._document_fingerprint = fingerprint
        self._temporary = tempfile.TemporaryDirectory(prefix="corporatebench-")
        self._temporary_path = Path(self._temporary.name)
        self._store: Store | None = None

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def document_fingerprint(self) -> str:
        return self._document_fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def document_names(self) -> tuple[str, ...]:
        return tuple(name.removeprefix("documents/") for name in self._documents)

    def close(self) -> None:
        self._archive.close()
        if self._store is not None:
            self._store = None
            gc.collect()
        self._temporary.cleanup()

    def query_sparql(self, query: str) -> str:
        query = self._validate_query(query)
        store = self._load_store()
        result = store.query(query)

        if not hasattr(result, "variables"):
            return json.dumps({"result": bool(result)})

        variables = [
            (variable, getattr(variable, "value", str(variable).lstrip("?")))
            for variable in result.variables
        ]
        rows: list[dict[str, Any]] = []
        for index, solution in enumerate(result):
            if index >= self.MAX_RESULT_ROWS:
                break
            row: dict[str, Any] = {}
            for variable, name in variables:
                value = solution[variable]
                if value is not None:
                    row[name] = _term_value(value)
            rows.append(row)

        output = json.dumps({"rows": rows}, ensure_ascii=False)
        if len(output) > self.MAX_RESULT_CHARS:
            return json.dumps({"error": "Result is too large; narrow the query."})
        return output

    def document_for_embedding(self, index: int) -> str:
        filename, content = self.read_document(index)
        content = content[: self.MAX_EMBEDDING_CONTENT_CHARS]
        return f"path: {filename}\n\ncontent: {content}"

    def read_document(self, index: int) -> tuple[str, str]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Document index must be an integer")
        if not 0 <= index < self.document_count:
            raise IndexError("Document index is out of range")
        name = self._documents[index]
        info = self._members[name]
        if info.file_size > self.MAX_DOCUMENT_BYTES:
            raise ValueError("KB document is too large")
        content = self._archive.read(name).decode("utf-8", errors="replace")
        return name.removeprefix("documents/"), content

    def render_documents(self, indices: list[int]) -> str:
        if not 1 <= len(indices) <= 50 or len(indices) != len(set(indices)):
            raise ValueError("Invalid document result set")
        sections: list[str] = []
        size = 0
        for index in indices:
            filename, content = self.read_document(index)
            section = f"# {filename}\n\n{content}"
            remaining = self.MAX_SEARCH_RESULT_CHARS - size
            if remaining <= 0:
                break
            sections.append(section[:remaining])
            size += len(sections[-1])
        return "\n\n".join(sections)

    def iter_graph_quads(self):
        with self._archive.open("graph.nq") as graph:
            yield from parse_rdf(graph, RdfFormat.N_QUADS)

    def iter_ontology_quads(self):
        for name in sorted(self._members):
            if name.startswith("ontologies/") and name.endswith(".ttl"):
                with self._archive.open(name) as ontology:
                    yield from parse_rdf(ontology, RdfFormat.TURTLE)

    def _validate_archive(self) -> dict[str, zipfile.ZipInfo]:
        infos = self._archive.infolist()
        if not 1 <= len(infos) <= self.MAX_MEMBERS:
            raise ValueError("KB archive has an invalid member count")

        total_size = 0
        members: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            name = info.filename
            path = PurePosixPath(name)
            mode = info.external_attr >> 16
            if (
                not name
                or "\\" in name
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or stat.S_ISLNK(mode)
            ):
                raise ValueError("KB archive contains an unsafe member")
            if info.file_size > self.MAX_MEMBER_BYTES:
                raise ValueError("KB archive member is too large")
            if info.file_size and not info.compress_size:
                raise ValueError("KB archive has an invalid compression ratio")
            if (
                info.compress_size
                and info.file_size / info.compress_size > self.MAX_COMPRESSION_RATIO
            ):
                raise ValueError("KB archive has an unsafe compression ratio")
            total_size += info.file_size
            if total_size > self.MAX_UNCOMPRESSED_BYTES:
                raise ValueError("KB archive is too large")
            if not info.is_dir():
                if name in members:
                    raise ValueError("KB archive contains a duplicate member")
                members[name] = info

        if "graph.nq" not in members:
            raise ValueError("KB archive is missing graph.nq")
        if not any(name.startswith("documents/") for name in members):
            raise ValueError("KB archive contains no documents")
        return members

    def _load_store(self) -> Store:
        if self._store is not None:
            return self._store

        store = Store(str(self._temporary_path / "oxigraph"))
        with self._archive.open("graph.nq") as graph:
            store.load(graph, RdfFormat.N_QUADS)
        for name in self._members:
            if name.startswith("ontologies/") and name.endswith(".ttl"):
                with self._archive.open(name) as ontology:
                    store.load(ontology, RdfFormat.TURTLE)
        self._store = store
        return store

    def _validate_query(self, query: str) -> str:
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > self.MAX_QUERY_CHARS
            or "\x00" in query
        ):
            raise ValueError("Invalid SPARQL query")
        operations = [match.upper() for match in _QUERY_OPERATIONS.findall(query)]
        if not operations or operations[0] not in {"SELECT", "ASK"}:
            raise ValueError("Only SPARQL SELECT and ASK queries are allowed")
        if any(operation in _FORBIDDEN_OPERATIONS for operation in operations):
            raise ValueError("SPARQL query contains a forbidden operation")
        return query.strip()


def _term_value(term: Any) -> str:
    value = getattr(term, "value", str(term))
    if isinstance(value, str) and value.startswith(_ENTITY_PREFIX):
        return value.removeprefix(_ENTITY_PREFIX)
    return str(value)
