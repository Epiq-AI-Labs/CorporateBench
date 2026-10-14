from __future__ import annotations

import ast
import hashlib
import os
import re
import sys
import tempfile
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from itertools import islice
from pathlib import Path
from threading import Timer
from typing import Any

import duckdb
from filelock import FileLock
from pyoxigraph import Literal, NamedNode, Triple

from .kb import KnowledgeBase

_ENTITY_PREFIX = "http://storybeat.company/entity/"
_CORE = "https://storybeat.company/ontology/core#"
_COMPANY = "https://storybeat.company/ontology/company#"
_PROJECTS = "https://storybeat.company/ontology/projects#"
_MEETINGS = "https://storybeat.company/ontology/meetings#"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_ENTITY_TYPES = {
    "company": f"{_COMPANY}Company",
    "department": f"{_COMPANY}Department",
    "team": f"{_COMPANY}Team",
    "employee": f"{_COMPANY}Employee",
    "task": f"{_PROJECTS}Task",
    "project": f"{_PROJECTS}Project",
    "meeting": f"{_MEETINGS}Meeting",
}
_ENTITY_TABLES = frozenset(f"gold_{name}" for name in _ENTITY_TYPES)
_KB_TABLES = _ENTITY_TABLES | {"gold_relationship"}
_DOCUMENT_TABLE = "gold_document"
_ALL_TABLES = _KB_TABLES | {_DOCUMENT_TABLE}
_TASK_TABLES = {
    "kb_qa": _KB_TABLES,
    "topic_qa": frozenset({_DOCUMENT_TABLE}),
    "integrated_qa": _ALL_TABLES,
}
_TOPIC_LEVELS = {"zenith": 0, "streamvibe": 1, "biocure": 2, "pound": 3}

_FORBIDDEN_SQL = re.compile(
    r"\b(ATTACH|CALL|COPY|CREATE|DELETE|DETACH|DROP|EXPORT|IMPORT|INSERT|"
    r"INSTALL|LOAD|PRAGMA|RESET|SET|TRUNCATE|UPDATE|VACUUM|ALTER|"
    r"CURRENT_SETTING|GETENV|GLOB|READ_BLOB|READ_CSV|READ_JSON|READ_PARQUET|"
    r"READ_TEXT|DUCKDB_[A-Z0-9_]*|INFORMATION_SCHEMA|PG_CATALOG|SQLITE_[A-Z0-9_]*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SqlIndexInfo:
    path: Path
    company: str
    entities: int
    relationships: int
    documents: int


class SqlKnowledgeBase:
    VERSION = 2
    MAX_QUERY_CHARS = 20_000
    MAX_RESULT_ROWS = 100_000
    MAX_RESULT_CHARS = 10_000_000
    QUERY_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        company: str,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if company not in _TOPIC_LEVELS:
            raise ValueError("Unknown company")
        default_cache = Path.home() / ".cache" / "corporatebench" / "sql"
        self.cache_dir = _prepare_cache_dir(
            default_cache if cache_dir is None else cache_dir
        )
        self.kb = kb
        self.company = company
        key = hashlib.sha256(
            f"{self.VERSION}:{company}:{kb.fingerprint}".encode()
        ).hexdigest()[:24]
        self.path = self.cache_dir / f"{company}-{key}.duckdb"
        if not self.path.resolve(strict=False).is_relative_to(self.cache_dir):
            raise ValueError("SQL cache path is unsafe")

    def bootstrap(self) -> SqlIndexInfo:
        lock_path = self.cache_dir / f".{self.path.name}.lock"
        if lock_path.is_symlink():
            raise ValueError("SQL cache lock is unsafe")
        with FileLock(lock_path, timeout=3_600):
            if self.path.exists():
                if self.path.is_symlink() or not self.path.is_file():
                    raise ValueError("SQL cache file is unsafe")
                return self._inspect()
            self._build()
            return self._inspect()

    def execute(self, query: str, *, task: str) -> str:
        try:
            self.bootstrap()
            query = self._validate_query(query, task)
            with self._connect(read_only=True) as connection:
                tables = {
                    _table_name(name)
                    for name in connection.get_table_names(query, qualified=True)
                }
                if not tables.issubset(_TASK_TABLES[task]):
                    raise ValueError("Query accesses tables outside this task")
                statements = connection.extract_statements(query)
                if (
                    len(statements) != 1
                    or getattr(getattr(statements[0], "type", None), "name", "")
                    != "SELECT"
                ):
                    raise ValueError("Only one SELECT query is allowed")

                timer = Timer(self.QUERY_TIMEOUT_SECONDS, connection.interrupt)
                timer.daemon = True
                timer.start()
                try:
                    cursor = connection.execute(query)
                    rows = cursor.fetchmany(self.MAX_RESULT_ROWS + 1)
                    columns = [description[0] for description in cursor.description]
                finally:
                    timer.cancel()
            if len(rows) > self.MAX_RESULT_ROWS:
                return "Error executing query: Result is too large; narrow the query."
            return _format_rows(columns, rows, max_chars=self.MAX_RESULT_CHARS)
        except Exception as error:
            return f"Error executing query: {_safe_error(error)}"

    def _validate_query(self, query: str, task: str) -> str:
        if task not in _TASK_TABLES:
            raise ValueError("Unknown QA task")
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > self.MAX_QUERY_CHARS
            or "\x00" in query
            or "--" in query
            or "/*" in query
            or "*/" in query
        ):
            raise ValueError("Invalid SQL query")
        stripped = query.strip()
        if re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE) is None:
            raise ValueError("Only SELECT queries are allowed")
        if _FORBIDDEN_SQL.search(stripped):
            raise ValueError("SQL query contains a forbidden operation")
        return stripped

    def _build(self) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.cache_dir, prefix=f".{self.company}-", suffix=".duckdb"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        try:
            entities, documents, relationships = _extract_records(
                self.kb, topic_level=_TOPIC_LEVELS[self.company]
            )
            with duckdb.connect(str(temporary), config={"threads": "1"}) as connection:
                _create_schema(connection)
                _insert_entities(connection, entities)
                _insert_documents(connection, documents)
                relationship_count = _insert_relationships(
                    connection, _relationship_records(relationships)
                )
                connection.execute(
                    "INSERT INTO corporatebench_meta VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        self.VERSION,
                        self.company,
                        self.kb.fingerprint,
                        len(entities),
                        relationship_count,
                        len(documents),
                    ],
                )
                connection.execute("CHECKPOINT")
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _inspect(self) -> SqlIndexInfo:
        with self._connect(read_only=True) as connection:
            row = connection.execute(
                "SELECT version, company, fingerprint, entities, relationships, "
                "documents FROM corporatebench_meta"
            ).fetchone()
        if (
            row is None
            or row[0] != self.VERSION
            or row[1] != self.company
            or row[2] != self.kb.fingerprint
        ):
            raise ValueError("SQL cache does not match the benchmark data")
        return SqlIndexInfo(
            path=self.path,
            company=self.company,
            entities=int(row[3]),
            relationships=int(row[4]),
            documents=int(row[5]),
        )

    def _connect(self, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(
            str(self.path),
            read_only=read_only,
            config={
                "threads": "1",
                "memory_limit": "512MB",
                "enable_external_access": "false",
                "allow_unsigned_extensions": "false",
                "allow_community_extensions": "false",
            },
        )


def _create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE corporatebench_meta (
            version INTEGER, company VARCHAR, fingerprint VARCHAR,
            entities INTEGER, relationships INTEGER, documents INTEGER
        );
        CREATE TABLE gold_company (
            id VARCHAR PRIMARY KEY, type TEXT, name TEXT, aliases TEXT[]
        );
        CREATE TABLE gold_department (
            id VARCHAR PRIMARY KEY, type TEXT, name TEXT, aliases TEXT[]
        );
        CREATE TABLE gold_team (
            id VARCHAR PRIMARY KEY, type TEXT, name TEXT, aliases TEXT[]
        );
        CREATE TABLE gold_employee (
            id VARCHAR PRIMARY KEY, type TEXT, name TEXT, aliases TEXT[],
            title TEXT, personal_email TEXT, work_email TEXT
        );
        CREATE TABLE gold_task (
            id VARCHAR PRIMARY KEY, type TEXT, name TEXT, aliases TEXT[],
            description TEXT, status TEXT
        );
        CREATE TABLE gold_project (
            id VARCHAR PRIMARY KEY, type TEXT, name TEXT, aliases TEXT[]
        );
        CREATE TABLE gold_meeting (
            id VARCHAR PRIMARY KEY, type TEXT, title TEXT, aliases TEXT[],
            meeting_type TEXT, topic TEXT, date DATE
        );
        CREATE TABLE gold_relationship (
            id VARCHAR PRIMARY KEY, type VARCHAR NOT NULL, source VARCHAR NOT NULL,
            target VARCHAR NOT NULL, temporality VARCHAR NOT NULL,
            beginning DATE, "end" DATE
        );
        CREATE TABLE gold_document (
            id VARCHAR PRIMARY KEY, date DATE, topic TEXT, sender_alias TEXT,
            sender_name TEXT, sender_email TEXT, recipient_aliases TEXT[],
            recipient_names TEXT[], recipient_emails TEXT[],
            represents_meeting VARCHAR
        );
        """
    )


def _extract_records(
    kb: KnowledgeBase,
    *,
    topic_level: int,
    include_documents: bool = True,
    relationship_types: Collection[str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, str], tuple[str | None, str | None, bool]],
]:
    type_names = {uri: name for name, uri in _ENTITY_TYPES.items()}
    if include_documents:
        type_names[f"{_CORE}Document"] = "document"
    subjects: dict[str, str] = {}
    relationship_predicates = set(_relationship_predicates(kb))
    entity_properties = {
        "legalName",
        "acronym",
        "name",
        "hasName",
        "hasAlias",
        "alias",
        "jobTitle",
        "personalEmail",
        "workEmail",
        "description",
        "status",
        "workStatus",
        "title",
        "meetingType",
        "topic",
        "date",
    }
    document_properties = {
        "filename",
        "date",
        "topic",
        "firstParentTopic",
        "secondParentTopic",
        "thirdParentTopic",
        "fourthParentTopic",
        "senderEntityAlias",
        "senderEntityName",
        "senderEntityEmail",
        "recipientEntityAlias",
        "recipientEntityName",
        "recipientEntityEmail",
        "representsMeeting",
    }
    wanted_properties = entity_properties | (
        document_properties if include_documents else set()
    )
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    relationships: dict[tuple[str, str, str], tuple[str | None, str | None, bool]] = {}
    for quad in kb.iter_graph_quads():
        if (
            isinstance(quad.subject, NamedNode)
            and quad.predicate.value == _RDF_TYPE
            and isinstance(quad.object, NamedNode)
        ):
            if quad.object.value in type_names:
                subjects[sys.intern(quad.subject.value)] = type_names[quad.object.value]
            elif quad.object.value == f"{_CORE}Relationship":
                relationship_predicates.add(quad.subject.value)

        if isinstance(quad.subject, NamedNode) and isinstance(quad.object, Literal):
            property_name = _local_name(quad.predicate.value)
            if property_name in wanted_properties:
                subject = sys.intern(quad.subject.value)
                grouped[subject][property_name].append(quad.object.value)

        if (
            isinstance(quad.subject, NamedNode)
            and isinstance(quad.object, NamedNode)
            and quad.predicate.value in relationship_predicates
        ):
            relationship_type = _local_name(quad.predicate.value)
            if (
                relationship_types is not None
                and relationship_type not in relationship_types
            ):
                continue
            key = _relationship_key(
                quad.subject.value, quad.predicate.value, quad.object.value
            )
            beginning, end, _ = relationships.get(key, (None, None, False))
            relationships[key] = (beginning, end, True)
        elif (
            isinstance(quad.subject, Triple)
            and isinstance(quad.subject.subject, NamedNode)
            and isinstance(quad.subject.object, NamedNode)
            and quad.subject.predicate.value in relationship_predicates
            and isinstance(quad.object, Literal)
            and quad.predicate.value in {f"{_CORE}beginning", f"{_CORE}end"}
        ):
            relationship_type = _local_name(quad.subject.predicate.value)
            if (
                relationship_types is not None
                and relationship_type not in relationship_types
            ):
                continue
            key = _relationship_key(
                quad.subject.subject.value,
                quad.subject.predicate.value,
                quad.subject.object.value,
            )
            beginning, end, exists = relationships.get(key, (None, None, False))
            if quad.predicate.value == f"{_CORE}beginning":
                beginning = quad.object.value
            else:
                end = quad.object.value
            relationships[key] = (beginning, end, exists)

    for properties in grouped.values():
        for values in properties.values():
            values.sort()

    entities = [
        _map_entity(_entity_id(subject), entity_type, grouped[subject])
        for subject, entity_type in sorted(subjects.items())
        if entity_type != "document"
    ]
    documents = (
        _map_documents(
            (
                grouped[subject]
                for subject, entity_type in sorted(subjects.items())
                if entity_type == "document"
            ),
            topic_level=topic_level,
        )
        if include_documents
        else []
    )
    return entities, documents, relationships


def _relationship_predicates(kb: KnowledgeBase) -> frozenset[str]:
    return frozenset(
        quad.subject.value
        for quad in kb.iter_ontology_quads()
        if isinstance(quad.subject, NamedNode)
        and quad.predicate.value == _RDF_TYPE
        and isinstance(quad.object, NamedNode)
        and quad.object.value == f"{_CORE}Relationship"
    )


def _relationship_key(
    source: str, relationship: str, target: str
) -> tuple[str, str, str]:
    return (
        sys.intern(_entity_id(source)),
        sys.intern(_local_name(relationship)),
        sys.intern(_entity_id(target)),
    )


def _map_entity(
    entity_id: str, entity_type: str, properties: dict[str, list[str]]
) -> dict[str, Any]:
    common = {"id": entity_id, "type": entity_type.capitalize()}
    if entity_type == "company":
        return {
            **common,
            "name": _one(properties, "legalName"),
            "aliases": _many(properties, "acronym"),
        }
    if entity_type in {"department", "team"}:
        return {
            **common,
            "name": _one(properties, "name"),
            "aliases": _many(properties, "acronym"),
        }
    if entity_type == "employee":
        return {
            **common,
            "name": _one(properties, "hasName", "name"),
            "aliases": _many(properties, "hasAlias", "alias"),
            "title": _one(properties, "jobTitle"),
            "personal_email": _one(properties, "personalEmail"),
            "work_email": _one(properties, "workEmail"),
        }
    if entity_type == "task":
        return {
            **common,
            "name": _one(properties, "name"),
            "aliases": _many(properties, "alias"),
            "description": _one(properties, "description"),
            "status": _one(properties, "status", "workStatus"),
        }
    if entity_type == "project":
        return {
            **common,
            "name": _one(properties, "name"),
            "aliases": _many(properties, "alias"),
        }
    return {
        "id": entity_id,
        "type": entity_type.capitalize(),
        "title": _one(properties, "title"),
        "aliases": _many(properties, "alias", "hasAlias", "acronym"),
        "meeting_type": _one(properties, "meetingType"),
        "topic": _one(properties, "topic"),
        "date": _date(_one(properties, "date")),
    }


def _map_documents(
    property_sets: Iterable[dict[str, list[str]]], *, topic_level: int
) -> list[dict[str, Any]]:
    parent = {
        0: "topic",
        1: "firstParentTopic",
        2: "secondParentTopic",
        3: "thirdParentTopic",
        4: "fourthParentTopic",
    }[topic_level]
    documents: list[dict[str, Any]] = []
    for properties in property_sets:
        filename = _one(properties, "filename")
        if not filename:
            continue
        documents.append(
            {
                "id": filename,
                "date": _date(_one(properties, "date")),
                "topic": _one(properties, parent) or _one(properties, "topic"),
                "sender_alias": _one(properties, "senderEntityAlias"),
                "sender_name": _one(properties, "senderEntityName"),
                "sender_email": _one(properties, "senderEntityEmail"),
                "recipient_aliases": _array(_one(properties, "recipientEntityAlias")),
                "recipient_names": _array(_one(properties, "recipientEntityName")),
                "recipient_emails": _array(_one(properties, "recipientEntityEmail")),
                "represents_meeting": _one(properties, "representsMeeting"),
            }
        )
    return documents


def _relationship_records(
    relationships: dict[tuple[str, str, str], tuple[str | None, str | None, bool]],
) -> Iterator[dict[str, Any]]:
    while relationships:
        (source, relationship_type, target), (beginning, end, exists) = (
            relationships.popitem()
        )
        if not exists:
            continue
        identity = "\x00".join(
            (source, relationship_type, target, beginning or "", end or "")
        )
        yield {
            "id": "relationship_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            "type": relationship_type,
            "source": source,
            "target": target,
            "temporality": "ongoing" if not end else "bounded",
            "beginning": _date(beginning),
            "end": _date(end),
        }


def _insert_entities(
    connection: duckdb.DuckDBPyConnection, records: list[dict[str, Any]]
) -> None:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["type"].casefold()].append(record)
    columns = {
        "company": ("id", "type", "name", "aliases"),
        "department": ("id", "type", "name", "aliases"),
        "team": ("id", "type", "name", "aliases"),
        "employee": (
            "id",
            "type",
            "name",
            "aliases",
            "title",
            "personal_email",
            "work_email",
        ),
        "task": ("id", "type", "name", "aliases", "description", "status"),
        "project": ("id", "type", "name", "aliases"),
        "meeting": (
            "id",
            "type",
            "title",
            "aliases",
            "meeting_type",
            "topic",
            "date",
        ),
    }
    for entity_type, entity_columns in columns.items():
        rows = by_type[entity_type]
        if rows:
            types = tuple(
                "VARCHAR[]"
                if column == "aliases"
                else "DATE"
                if column == "date"
                else "VARCHAR"
                for column in entity_columns
            )
            _bulk_insert(
                connection,
                f"gold_{entity_type}",
                entity_columns,
                types,
                rows,
            )


def _insert_documents(
    connection: duckdb.DuckDBPyConnection, records: list[dict[str, Any]]
) -> None:
    columns = (
        "id",
        "date",
        "topic",
        "sender_alias",
        "sender_name",
        "sender_email",
        "recipient_aliases",
        "recipient_names",
        "recipient_emails",
        "represents_meeting",
    )
    if records:
        types = (
            "VARCHAR",
            "DATE",
            "VARCHAR",
            "VARCHAR",
            "VARCHAR",
            "VARCHAR",
            "VARCHAR[]",
            "VARCHAR[]",
            "VARCHAR[]",
            "VARCHAR",
        )
        _bulk_insert(connection, "gold_document", columns, types, records)


def _insert_relationships(
    connection: duckdb.DuckDBPyConnection, records: Iterable[dict[str, Any]]
) -> int:
    columns = ("id", "type", "source", "target", "temporality", "beginning", "end")
    types = ("VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR", "DATE", "DATE")
    return _bulk_insert(connection, "gold_relationship", columns, types, records)


def _bulk_insert(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: tuple[str, ...],
    types: tuple[str, ...],
    records: Iterable[dict[str, Any]],
    *,
    batch_size: int = 10_000,
) -> int:
    if len(columns) != len(types):
        raise ValueError("Invalid bulk insert schema")
    expressions = ", ".join(f"unnest(?::{column_type}[])" for column_type in types)
    query = f"INSERT INTO {table} SELECT {expressions}"
    total = 0
    iterator = iter(records)
    while batch := list(islice(iterator, batch_size)):
        parameters = [[record.get(column) for record in batch] for column in columns]
        connection.execute(query, parameters)
        total += len(batch)
    return total


def _format_rows(columns: list[str], rows: list[tuple], *, max_chars: int) -> str:
    if not rows:
        return "No results found."
    header = " | ".join(f"{column:<20}" for column in columns)
    lines = [header, "-" * len(header)]
    size = sum(len(line) + 1 for line in lines)
    for row in rows:
        line = " | ".join(f"{str(value):<20}" for value in row)
        size += len(line) + 1
        if size > max_chars:
            return "Error executing query: Result is too large; narrow the query."
        lines.append(line)
    return "\n".join(lines)


def _table_name(reference: str) -> str:
    base = re.split(r"\s+AS\s+", reference, maxsplit=1, flags=re.IGNORECASE)[0]
    name = base.rsplit(".", 1)[-1].strip('"').casefold()
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError("Invalid table reference")
    return name


def _local_name(uri: str) -> str:
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _entity_id(uri: str) -> str:
    return uri.removeprefix(_ENTITY_PREFIX)


def _one(properties: dict[str, list[str]], *names: str) -> str | None:
    for name in names:
        values = properties.get(name)
        if values:
            return values[0]
    return None


def _many(properties: dict[str, list[str]], *names: str) -> list[str] | None:
    for name in names:
        values = properties.get(name)
        if values:
            return list(dict.fromkeys(values))
    return None


def _array(value: str | None) -> list[str] | None:
    if not value:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    if isinstance(parsed, list):
        values = [str(item) for item in parsed]
    else:
        values = [str(parsed)]
    return values or None


def _date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _safe_error(error: Exception) -> str:
    message = str(error).splitlines()[0]
    message = re.sub(r"[\x00-\x1f\x7f]", " ", message).strip()
    return (message or "Query failed")[:500]


def _prepare_cache_dir(path: str | os.PathLike[str]) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("sql_cache_dir is invalid")
    candidate = Path(raw).expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("sql_cache_dir may not be a symlink")
    candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("sql_cache_dir must be a directory")
    return resolved
