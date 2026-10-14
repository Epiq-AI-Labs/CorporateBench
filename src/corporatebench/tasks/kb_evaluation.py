from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from rapidfuzz import fuzz
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from ..extraction_prompt import extraction_prompt
from ..kb import KnowledgeBase
from ..metrics import MetricsByType, metrics
from ..sql import _TOPIC_LEVELS, _extract_records
from ..types import ModelBackend, ModelRun

TASK_NAME = "kb_evaluation"
TASK_LABEL = "KB Evaluation"

_ENTITY_TYPES = frozenset(
    {"Employee", "Company", "Department", "Team", "Task", "Project", "Meeting"}
)
_RELATIONSHIP_TYPES = frozenset(
    {
        "worksOn",
        "worksAt",
        "reportsTo",
        "memberOf",
        "departmentBelongsToCompany",
        "teamBelongsToDepartment",
        "organize",
        "attend",
        "mentionsTask",
        "mentionsProject",
        "mentionsEmployee",
        "mentionsTeam",
        "mentionsDepartment",
    }
)
_EVALUATED_RELATIONSHIP_TYPES = frozenset(
    {
        "worksOn",
        "reportsTo",
        "memberOf",
        "worksAt",
        "attend",
        "organize",
        "departmentBelongsToCompany",
        "teamBelongsToDepartment",
    }
)
_RELATIONSHIP_SIGNATURES = {
    "worksOn": ({"Employee"}, {"Task"}),
    "worksAt": ({"Employee"}, {"Company"}),
    "reportsTo": ({"Employee"}, {"Employee"}),
    "memberOf": ({"Employee"}, {"Team", "Department"}),
    "departmentBelongsToCompany": ({"Department"}, {"Company"}),
    "teamBelongsToDepartment": ({"Team"}, {"Department"}),
    "organize": ({"Employee"}, {"Meeting"}),
    "attend": ({"Employee"}, {"Meeting"}),
    "mentionsTask": ({"Meeting"}, {"Task"}),
    "mentionsProject": ({"Meeting"}, {"Project"}),
    "mentionsEmployee": ({"Meeting"}, {"Employee"}),
    "mentionsTeam": ({"Meeting"}, {"Team"}),
    "mentionsDepartment": ({"Meeting"}, {"Department"}),
}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_MAX_PREDICTED_ENTITIES = 300_000
_MAX_ENTITY_MENTIONS = 1_000_000
_MAX_PREDICTED_RELATIONSHIPS = 750_000
_EndpointRef = tuple[str, str]
_AggregatedRelationships = dict[
    tuple[str, _EndpointRef, _EndpointRef], tuple[set[str], set[str]]
]


def _strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _entity_schema(entity_type: str, fields: tuple[str, ...]) -> dict[str, Any]:
    properties: dict[str, Any] = {"type": {"type": "string", "enum": [entity_type]}}
    for field_name in fields:
        properties[field_name] = (
            {"type": "array", "items": {"type": "string"}}
            if field_name in {"aliases", "assigned_to"}
            else {"type": "string"}
        )
    return _strict_object(properties)


_EXTRACTION_SCHEMA = _strict_object(
    {
        "document_type": {
            "type": "string",
            "enum": ["agenda", "meeting_minute", "regular_email"],
        },
        "entities": {
            "type": "array",
            "items": {
                "anyOf": [
                    _entity_schema(
                        "Employee",
                        ("name", "email", "aliases", "title", "department", "team"),
                    ),
                    _entity_schema("Company", ("name", "aliases")),
                    _entity_schema("Department", ("name", "aliases")),
                    _entity_schema("Team", ("name", "aliases", "department")),
                    _entity_schema(
                        "Task",
                        (
                            "name",
                            "aliases",
                            "description",
                            "status",
                            "assigned_to",
                            "project",
                        ),
                    ),
                    _entity_schema("Project", ("name", "aliases", "description")),
                    _entity_schema(
                        "Meeting",
                        (
                            "title",
                            "aliases",
                            "date",
                            "document_type",
                            "meeting_type",
                        ),
                    ),
                ]
            },
        },
        "relationships": {
            "type": "array",
            "items": _strict_object(
                {
                    "type": {
                        "type": "string",
                        "enum": sorted(_RELATIONSHIP_TYPES),
                    },
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "context": {"type": "string"},
                    "mentions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "temporality": {
                        "type": "string",
                        "enum": ["start", "end", "ongoing"],
                    },
                    "beginning": {"type": "string"},
                    "end": {"type": "string"},
                }
            ),
        },
    }
)


@dataclass(frozen=True, slots=True)
class KBEvaluationResult:
    company: str
    model: str
    documents: int
    model_errors: int
    failed_documents: tuple[str, ...]
    entities: MetricsByType
    relationships: MetricsByType
    temporal_relationships: MetricsByType

    def to_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "model": self.model,
            "documents": self.documents,
            "model_errors": self.model_errors,
            "failed_documents": list(self.failed_documents),
            "entities": self.entities.to_dict(),
            "relationships": self.relationships.to_dict(),
            "temporal_relationships": self.temporal_relationships.to_dict(),
        }


@dataclass(slots=True)
class _Entity:
    id: str
    type: str
    mentions: set[str]
    title: str = ""
    date: str = ""
    meeting_type: str = ""
    keys: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _Relationship:
    type: str
    source: str
    target: str
    beginning: str
    end: str
    source_ref: tuple[str, str] | None = None
    target_ref: tuple[str, str] | None = None


async def run_kb_evaluation(
    kb: KnowledgeBase,
    *,
    company: str,
    model: ModelBackend,
    concurrency: int,
) -> KBEvaluationResult:
    indices = [
        index
        for index, name in enumerate(kb.document_names)
        if not name.casefold().startswith("calendar")
    ]
    semaphore = asyncio.Semaphore(concurrency)

    async def extract(
        index: int,
    ) -> tuple[str, list[_Entity], list[_Relationship], bool]:
        filename = kb.document_names[index]
        try:
            async with semaphore:
                _, content = kb.read_document(index)
                prompt = extraction_prompt(filename, content)
                del content
                structured_run = getattr(model, "run_structured", None)
                if callable(structured_run):
                    response = await structured_run(
                        system="", prompt=prompt, schema=_EXTRACTION_SCHEMA
                    )
                else:
                    response = await model.run(
                        system="",
                        prompt=prompt,
                        tools=(),
                        max_tool_calls=0,
                    )
            if not isinstance(response, ModelRun):
                raise TypeError("model backend returned an invalid result")
            entities, relationships = _parse_response(response.answer)
            return filename, entities, relationships, False
        except Exception:
            return filename, [], [], True

    clusters = _EntityClusters()
    raw_relationships: _AggregatedRelationships = {}
    model_errors = 0
    failed_documents: list[str] = []
    batch_size = concurrency
    for offset in range(0, len(indices), batch_size):
        responses = await asyncio.gather(
            *(extract(index) for index in indices[offset : offset + batch_size])
        )
        for filename, entities, relationships, error in responses:
            clusters.add_all(
                entities,
                max_entities=_MAX_PREDICTED_ENTITIES,
                max_mentions=_MAX_ENTITY_MENTIONS,
            )
            _aggregate_relationships(
                raw_relationships,
                relationships,
                max_relationships=_MAX_PREDICTED_RELATIONSHIPS,
            )
            model_errors += error
            if error:
                failed_documents.append(filename)

    predicted_entities, predicted_relationships = await asyncio.to_thread(
        _finalize_predictions, clusters, raw_relationships
    )
    entity_scores, relationship_scores, temporal_scores = await asyncio.to_thread(
        _score_predictions,
        kb,
        company,
        predicted_entities,
        predicted_relationships,
    )
    return KBEvaluationResult(
        company=company,
        model=model.name,
        documents=len(indices),
        model_errors=model_errors,
        failed_documents=tuple(failed_documents),
        entities=entity_scores,
        relationships=relationship_scores,
        temporal_relationships=temporal_scores,
    )


def _finalize_predictions(
    clusters: _EntityClusters,
    relationships: _AggregatedRelationships,
) -> tuple[list[_Entity], dict[tuple[str, str, str], tuple[set[str], set[str]]]]:
    entities = clusters.entities()
    resolved = _resolve_aggregated_relationships(
        relationships, _mention_index(entities), _key_index(entities)
    )
    return entities, resolved


def _score_predictions(
    kb: KnowledgeBase,
    company: str,
    predicted_entities: list[_Entity],
    predicted_relationships: dict[tuple[str, str, str], tuple[set[str], set[str]]],
) -> tuple[MetricsByType, MetricsByType, MetricsByType]:
    ground_entities, _, ground_relationships = _extract_records(
        kb,
        topic_level=_TOPIC_LEVELS[company],
        include_documents=False,
        relationship_types=_EVALUATED_RELATIONSHIP_TYPES,
    )
    expected_entities = [_ground_entity(entity) for entity in ground_entities]
    entity_scores, gt_to_pred = _entity_metrics(expected_entities, predicted_entities)
    relationship_scores, temporal_scores = _relationship_metrics(
        ground_relationships,
        predicted_relationships,
        gt_to_pred,
    )
    return entity_scores, relationship_scores, temporal_scores


def _parse_response(answer: str) -> tuple[list[_Entity], list[_Relationship]]:
    if not isinstance(answer, str) or not 1 <= len(answer) <= 1_000_000:
        raise ValueError("Invalid extraction response")
    payload = json.loads(_FENCE.sub("", answer.strip()))
    if not isinstance(payload, dict) or not set(payload) <= {
        "document_type",
        "entities",
        "relationships",
    }:
        raise ValueError("Invalid extraction response")
    if payload.get("document_type") not in {
        None,
        "agenda",
        "meeting_minute",
        "regular_email",
    }:
        raise ValueError("Invalid document type")
    raw_entities = payload.get("entities")
    raw_relationships = payload.get("relationships")
    if (
        not isinstance(raw_entities, list)
        or not isinstance(raw_relationships, list)
        or len(raw_entities) > 250
        or len(raw_relationships) > 1_000
    ):
        raise ValueError("Invalid extraction response")

    entities: list[_Entity] = []
    for index, raw in enumerate(raw_entities):
        if not isinstance(raw, dict) or raw.get("type") not in _ENTITY_TYPES:
            raise ValueError("Invalid extracted entity")
        entity_date = _valid_date(raw.get("date"))
        mentions = _entity_mentions(raw)
        if not mentions:
            raise ValueError("Extracted entity has no name")
        entities.append(
            _Entity(
                id=f"raw:{index}",
                type=raw["type"],
                mentions=mentions,
                title=_text(raw.get("title")),
                date=entity_date,
                meeting_type=_text(raw.get("meeting_type")),
                keys=_entity_keys(raw, entity_date),
                names=_entity_names(raw),
            )
        )

    relationships: list[_Relationship] = []
    for raw in raw_relationships:
        if (
            not isinstance(raw, dict)
            or raw.get("type") not in _RELATIONSHIP_TYPES
            or raw.get("temporality") not in {"start", "end", "ongoing"}
        ):
            raise ValueError("Invalid extracted relationship")
        source = _text(raw.get("source"))
        target = _text(raw.get("target"))
        if not source or not target:
            raise ValueError("Extracted relationship is missing an endpoint")
        beginning = _valid_date(raw.get("beginning"))
        end = _valid_date(raw.get("end"))
        temporality = raw["temporality"]
        if (
            temporality == "start"
            and (not beginning or end)
            or temporality == "end"
            and (beginning or not end)
            or temporality == "ongoing"
            and (beginning or end)
        ):
            raise ValueError("Invalid relationship temporality")
        source_types, target_types = _RELATIONSHIP_SIGNATURES[raw["type"]]
        relationships.append(
            _Relationship(
                raw["type"],
                source,
                target,
                beginning,
                end,
                _local_entity_ref(source, source_types, entities),
                _local_entity_ref(target, target_types, entities),
            )
        )
    return entities, relationships


def _local_entity_ref(
    value: str,
    allowed_types: set[str],
    entities: list[_Entity],
) -> tuple[str, str] | None:
    normalized = _normalize(value)
    candidates = [
        entity
        for entity in entities
        if entity.type in allowed_types and normalized in entity.mentions
    ]
    if not candidates:
        return None
    entity = candidates[0]
    if normalized in entity.keys:
        return entity.type, normalized
    if entity.keys:
        return entity.type, sorted(entity.keys)[0]
    return None


def _entity_mentions(raw: dict[str, Any]) -> set[str]:
    values = [raw.get("name"), raw.get("email")]
    if raw.get("type") == "Meeting":
        values.append(raw.get("title"))
    aliases = raw.get("aliases", [])
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list) or len(aliases) > 50:
        raise ValueError("Invalid entity aliases")
    values.extend(aliases)
    return {_normalize(_text(value)) for value in values if _text(value)}


def _entity_keys(raw: dict[str, Any], entity_date: str) -> set[str]:
    entity_type = raw["type"]
    if entity_type == "Meeting":
        title = _normalize(_text(raw.get("title")))
        return {f"{title}\x00{entity_date}"} if title else set()
    values = (_text(raw.get("name")), _text(raw.get("email")))
    return {_normalize(value) for value in values if value}


def _entity_names(raw: dict[str, Any]) -> set[str]:
    values = [raw.get("name")]
    if raw.get("type") == "Meeting":
        values.append(raw.get("title"))
    return {_normalize(_text(value)) for value in values if _text(value)}


def _deduplicate_entities(entities: list[_Entity]) -> list[_Entity]:
    clusters = _EntityClusters()
    clusters.add_all(entities)
    return clusters.entities()


class _EntityClusters:
    def __init__(self) -> None:
        self._next_id = 0
        self._clusters: dict[int, _Entity] = {}
        self._mentions: dict[tuple[str, str], int] = {}
        self._mention_count = 0

    def __len__(self) -> int:
        return len(self._clusters)

    @property
    def mention_count(self) -> int:
        return self._mention_count

    def add_all(
        self,
        entities: list[_Entity],
        *,
        max_entities: int | None = None,
        max_mentions: int | None = None,
    ) -> None:
        for entity in entities:
            matches = {
                self._mentions[(entity.type, key)]
                for key in entity.keys
                if (entity.type, key) in self._mentions
            }
            matched_mentions = sum(
                len(self._clusters[cluster_id].mentions) for cluster_id in matches
            )
            combined_mentions = set(entity.mentions)
            for cluster_id in matches:
                combined_mentions.update(self._clusters[cluster_id].mentions)
            resulting_entities = len(self._clusters) - len(matches) + 1
            resulting_mentions = (
                self._mention_count - matched_mentions + len(combined_mentions)
            )
            if (
                max_entities is not None
                and resulting_entities > max_entities
                or max_mentions is not None
                and resulting_mentions > max_mentions
            ):
                raise RuntimeError(
                    "KB extraction exceeded the prediction resource limit"
                )
            if matches:
                cluster_id = min(matches)
                cluster = self._clusters[cluster_id]
            else:
                cluster_id = self._next_id
                self._next_id += 1
                cluster = _Entity(
                    id="",
                    type=entity.type,
                    mentions=set(),
                    keys=set(),
                )
                self._clusters[cluster_id] = cluster
            previous_mentions = len(cluster.mentions) + sum(
                len(self._clusters[other_id].mentions)
                for other_id in matches - {cluster_id}
            )
            for other_id in sorted(matches - {cluster_id}):
                other = self._clusters.pop(other_id)
                cluster.mentions.update(other.mentions)
                cluster.title = cluster.title or other.title
                cluster.date = cluster.date or other.date
                cluster.meeting_type = cluster.meeting_type or other.meeting_type
                cluster.keys.update(other.keys)
                cluster.names.update(other.names)
            cluster.mentions.update(entity.mentions)
            cluster.keys.update(entity.keys)
            cluster.names.update(entity.names)
            cluster.title = cluster.title or entity.title
            cluster.date = cluster.date or entity.date
            cluster.meeting_type = cluster.meeting_type or entity.meeting_type
            self._mention_count += len(cluster.mentions) - previous_mentions
            for key in cluster.keys:
                self._mentions[(cluster.type, key)] = cluster_id

    def entities(self) -> list[_Entity]:
        entities = _merge_similar_entities(
            [self._clusters[key] for key in sorted(self._clusters)]
        )
        for index, entity in enumerate(entities):
            entity.id = f"pred:{index}"
        return entities


def _merge_similar_entities(entities: list[_Entity]) -> list[_Entity]:
    current = [_copy_entity(entity) for entity in entities]
    for stage, matcher in (
        ("partial", _partial_entity_match),
        ("fuzzy", _fuzzy_entity_match),
    ):
        merged: list[_Entity] = []
        for entity_type in sorted({entity.type for entity in current}):
            type_entities = [entity for entity in current if entity.type == entity_type]
            if entity_type == "Meeting":
                merged.extend(type_entities)
                continue
            candidates = _similarity_candidates(type_entities, stage)
            used: set[int] = set()
            for index, entity in enumerate(type_entities):
                if index in used:
                    continue
                group = [entity]
                used.add(index)
                for other_index in sorted(candidates[index]):
                    if other_index <= index:
                        continue
                    if other_index in used:
                        continue
                    other = type_entities[other_index]
                    if matcher(entity, other):
                        group.append(other)
                        used.add(other_index)
                merged.append(_merge_entities(group))
        current = merged
    return current


def _similarity_candidates(entities: list[_Entity], stage: str) -> list[set[int]]:
    token_sets: list[set[str]] = []
    index: dict[str, set[int]] = defaultdict(set)
    for entity_index, entity in enumerate(entities):
        tokens: set[str] = set()
        for name in entity.names:
            if stage == "partial":
                tokens.update(_character_ngrams(name, 3))
            else:
                tokens.update(_character_ngrams(name, 2))
                if entity.type != "Employee":
                    tokens.update(f"word:{word}" for word in name.split())
        token_sets.append(tokens)
        for token in tokens:
            index[token].add(entity_index)

    candidates = [set() for _ in entities]
    for entity_index, tokens in enumerate(token_sets):
        limit = 1 if stage == "partial" else 5
        rarest = sorted(tokens, key=lambda token: (len(index[token]), token))[:limit]
        for token in rarest:
            candidates[entity_index].update(index[token])
        for other_index in tuple(candidates[entity_index]):
            candidates[other_index].add(entity_index)
    return candidates


def _character_ngrams(value: str, size: int) -> set[str]:
    if len(value) <= size:
        return {value}
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def _copy_entity(entity: _Entity) -> _Entity:
    return _Entity(
        id=entity.id,
        type=entity.type,
        mentions=set(entity.mentions),
        title=entity.title,
        date=entity.date,
        meeting_type=entity.meeting_type,
        keys=set(entity.keys),
        names=set(entity.names),
    )


def _merge_entities(entities: list[_Entity]) -> _Entity:
    merged = _copy_entity(entities[0])
    for entity in entities[1:]:
        merged.mentions.update(entity.mentions)
        merged.keys.update(entity.keys)
        merged.names.update(entity.names)
        merged.title = merged.title or entity.title
        merged.date = merged.date or entity.date
        merged.meeting_type = merged.meeting_type or entity.meeting_type
    return merged


def _partial_entity_match(left: _Entity, right: _Entity) -> bool:
    for left_name in left.names:
        for right_name in right.names:
            if (
                (left_name in right_name or right_name in left_name)
                and len(left_name) >= len(right_name) * 0.5
                and len(right_name) >= len(left_name) * 0.5
            ):
                return True
    return False


def _fuzzy_entity_match(left: _Entity, right: _Entity) -> bool:
    thresholds = {
        "Company": 80,
        "Department": 80,
        "Team": 80,
        "Project": 75,
        "Task": 70,
        "Employee": 75,
    }
    threshold = thresholds.get(left.type, 75)
    for left_name in left.names:
        for right_name in right.names:
            scores = [fuzz.ratio(left_name, right_name)]
            if left.type != "Employee":
                scores.extend(
                    (
                        fuzz.token_sort_ratio(left_name, right_name),
                        fuzz.token_set_ratio(left_name, right_name),
                    )
                )
            if max(scores) >= threshold:
                return True
    return False


def _mention_index(entities: list[_Entity]) -> dict[str, list[_Entity]]:
    index: dict[str, list[_Entity]] = defaultdict(list)
    for entity in entities:
        for mention in entity.mentions:
            index[mention].append(entity)
    return index


def _key_index(entities: list[_Entity]) -> dict[tuple[str, str], _Entity]:
    return {(entity.type, key): entity for entity in entities for key in entity.keys}


def _resolve_relationships(
    relationships: list[_Relationship], mention_index: dict[str, list[_Entity]]
) -> dict[tuple[str, str, str], tuple[set[str], set[str]]]:
    aggregated: _AggregatedRelationships = {}
    _aggregate_relationships(aggregated, relationships)
    entities = {
        entity.id: entity
        for candidates in mention_index.values()
        for entity in candidates
    }
    return _resolve_aggregated_relationships(
        aggregated, mention_index, _key_index(list(entities.values()))
    )


def _aggregate_relationships(
    aggregated: _AggregatedRelationships,
    relationships: list[_Relationship],
    *,
    max_relationships: int | None = None,
) -> None:
    for relationship in relationships:
        if relationship.type not in _EVALUATED_RELATIONSHIP_TYPES:
            continue
        key = (
            relationship.type,
            relationship.source_ref or ("", _normalize(relationship.source)),
            relationship.target_ref or ("", _normalize(relationship.target)),
        )
        if (
            key not in aggregated
            and max_relationships is not None
            and len(aggregated) >= max_relationships
        ):
            raise RuntimeError("KB extraction exceeded the prediction resource limit")
        beginning, end = aggregated.setdefault(key, (set(), set()))
        if relationship.beginning:
            beginning.add(relationship.beginning)
        if relationship.end:
            end.add(relationship.end)


def _resolve_aggregated_relationships(
    relationships: _AggregatedRelationships,
    mention_index: dict[str, list[_Entity]],
    key_index: dict[tuple[str, str], _Entity],
) -> dict[tuple[str, str, str], tuple[set[str], set[str]]]:
    resolved: dict[tuple[str, str, str], tuple[set[str], set[str]]] = {}
    unresolved = 0
    for (relationship_type, source_ref, target_ref), dates in relationships.items():
        source_types, target_types = _RELATIONSHIP_SIGNATURES[relationship_type]
        source = _resolve_endpoint(source_ref, source_types, mention_index, key_index)
        target = _resolve_endpoint(target_ref, target_types, mention_index, key_index)
        if source is None:
            source = f"unresolved:{unresolved}"
            unresolved += 1
        if target is None:
            target = f"unresolved:{unresolved}"
            unresolved += 1
        key = (relationship_type, source, target)
        beginning, end = resolved.setdefault(key, (set(), set()))
        beginning.update(dates[0])
        end.update(dates[1])
    return resolved


def _resolve_endpoint(
    value: _EndpointRef,
    allowed_types: set[str],
    mention_index: dict[str, list[_Entity]],
    key_index: dict[tuple[str, str], _Entity],
) -> str | None:
    if value[0]:
        candidate = key_index.get(value)
        if candidate is not None and candidate.type in allowed_types:
            return candidate.id
        return None
    candidates = [
        entity
        for entity in mention_index.get(value[1], [])
        if entity.type in allowed_types
    ]
    return candidates[0].id if candidates else None


def _ground_entity(raw: dict[str, Any]) -> _Entity:
    values = [raw.get("name")]
    if raw.get("type") == "Meeting":
        values.append(raw.get("title"))
    values.extend(raw.get("aliases") or [])
    values.extend((raw.get("personal_email"), raw.get("work_email")))
    return _Entity(
        id=raw["id"],
        type=raw["type"],
        mentions={_normalize(value) for value in values if value},
        title=str(raw.get("title") or ""),
        date=str(raw.get("date") or ""),
        meeting_type=str(raw.get("meeting_type") or ""),
        names={
            _normalize(value) for value in (raw.get("name"), raw.get("title")) if value
        },
    )


def _entity_metrics(
    expected: list[_Entity], predicted: list[_Entity]
) -> tuple[MetricsByType, dict[str, str | None]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_id = {entity.id: entity for entity in predicted}
    exact_mentions: dict[tuple[str, str], set[str]] = defaultdict(set)
    trigram_mentions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entity in predicted:
        for mention in entity.mentions:
            exact_mentions[(entity.type, mention)].add(entity.id)
            for trigram in _trigrams(mention):
                trigram_mentions[(entity.type, trigram)].add(entity.id)
    candidate_scores: dict[str, dict[str, float]] = {}
    for ground in expected:
        candidate_ids: set[str] = set()
        fuzzy_counts: Counter[str] = Counter()
        for mention in ground.mentions:
            candidate_ids.update(exact_mentions[(ground.type, mention)])
            for trigram in _trigrams(mention):
                fuzzy_counts.update(trigram_mentions[(ground.type, trigram)])
        candidate_ids.update(
            entity_id
            for entity_id, _ in sorted(
                fuzzy_counts.items(), key=lambda item: (-item[1], item[0])
            )[:256]
        )
        scores: dict[str, float] = {}
        for candidate_id in sorted(candidate_ids):
            candidate = by_id[candidate_id]
            mention_score = _mention_score(candidate.mentions, ground.mentions)
            if not mention_score[0]:
                continue
            scores[candidate_id] = _entity_match_weight(
                ground, candidate, mention_score
            )
        candidate_scores[ground.id] = scores

    mapping = _maximum_entity_mapping(candidate_scores)
    matched_predicted = {
        predicted_id for predicted_id in mapping.values() if predicted_id is not None
    }
    for ground in expected:
        predicted_id = mapping[ground.id]
        if predicted_id is None:
            counts[ground.type][2] += 1
            continue
        counts[ground.type][0] += 1
        properties = ("title",) if ground.type == "Employee" else ()
        if ground.type == "Meeting":
            properties = ("title", "date", "meeting_type")
        for property_name in properties:
            expected_value = getattr(ground, property_name)
            predicted_value = getattr(by_id[predicted_id], property_name)
            if predicted_value == expected_value:
                counts[ground.type][0] += 1
            elif expected_value:
                counts[ground.type][2] += 1
            else:
                counts[ground.type][1] += 1
    for entity in predicted:
        if entity.id not in matched_predicted:
            counts[entity.type][1] += 1
    return _metrics_from_counts(counts), mapping


def _entity_match_weight(
    ground: _Entity,
    candidate: _Entity,
    mention_score: tuple[int, float],
) -> float:
    property_score = 0
    if ground.type == "Employee":
        property_score = int(candidate.title == ground.title)
    elif ground.type == "Meeting":
        property_score = sum(
            getattr(candidate, name) == getattr(ground, name)
            for name in ("title", "date", "meeting_type")
        )
        return (
            1e12 + property_score * 10_000 + mention_score[0] * 100 + mention_score[1]
        )
    return 1e12 + mention_score[0] * 10_000 + mention_score[1] * 100 + property_score


def _maximum_entity_mapping(
    candidate_scores: dict[str, dict[str, float]],
) -> dict[str, str | None]:
    mapping: dict[str, str | None] = dict.fromkeys(candidate_scores)
    candidate_to_ground: dict[str, set[str]] = defaultdict(set)
    for ground_id, candidates in candidate_scores.items():
        for candidate_id in candidates:
            candidate_to_ground[candidate_id].add(ground_id)

    remaining = {ground_id for ground_id, scores in candidate_scores.items() if scores}
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        ground_ids = {start}
        candidate_ids: set[str] = set()
        pending = [start]
        while pending:
            ground_id = pending.pop()
            for candidate_id in candidate_scores[ground_id]:
                if candidate_id in candidate_ids:
                    continue
                candidate_ids.add(candidate_id)
                for linked_ground in candidate_to_ground[candidate_id]:
                    if linked_ground not in ground_ids:
                        ground_ids.add(linked_ground)
                        remaining.discard(linked_ground)
                        pending.append(linked_ground)

        rows = sorted(ground_ids)
        columns = sorted(candidate_ids)
        column_index = {
            candidate_id: index for index, candidate_id in enumerate(columns)
        }
        matrix_rows: list[int] = []
        matrix_columns: list[int] = []
        matrix_values: list[float] = []
        for row, ground_id in enumerate(rows):
            for candidate_id, weight in candidate_scores[ground_id].items():
                matrix_rows.append(row)
                matrix_columns.append(column_index[candidate_id])
                matrix_values.append(-weight)
            matrix_rows.append(row)
            matrix_columns.append(len(columns) + row)
            matrix_values.append(1.0)
        matrix = coo_matrix(
            (matrix_values, (matrix_rows, matrix_columns)),
            shape=(len(rows), len(columns) + len(rows)),
        ).tocsr()
        row_indices, column_indices = min_weight_full_bipartite_matching(matrix)
        for row, column in zip(row_indices, column_indices, strict=True):
            if column < len(columns):
                mapping[rows[row]] = columns[column]
    return mapping


def _trigrams(value: str) -> set[str]:
    return {value[index : index + 3] for index in range(max(1, len(value) - 2))}


def _mention_score(predicted: set[str], expected: set[str]) -> tuple[int, float]:
    weights: list[float] = []
    matches = 0
    for prediction in predicted:
        best = 0.0
        for ground in expected:
            if prediction == ground:
                best = 1.0
                break
            if (
                (prediction in ground or ground in prediction)
                and len(prediction) >= len(ground) * 0.7
                and len(ground) >= len(prediction) * 0.7
            ):
                best = max(best, 0.85)
            threshold = 0.9 if "@" in prediction and "." in prediction else 0.7
            similarity = fuzz.ratio(prediction, ground) / 100
            if similarity >= threshold:
                best = max(best, 0.7)
        if best:
            matches += 1
            weights.append(best)
    return matches, sum(weights) / len(weights) if weights else 0.0


def _relationship_metrics(
    expected: dict[tuple[str, str, str], tuple[str | None, str | None, bool]],
    predicted: dict[tuple[str, str, str], tuple[set[str], set[str]]],
    gt_to_pred: dict[str, str | None],
) -> tuple[MetricsByType, MetricsByType]:
    structural: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    temporal: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    found: set[tuple[str, str, str]] = set()
    for (source, relation_type, target), (start, end, exists) in expected.items():
        if not exists or relation_type not in _EVALUATED_RELATIONSHIP_TYPES:
            continue
        predicted_source = gt_to_pred.get(source)
        predicted_target = gt_to_pred.get(target)
        key = (relation_type, predicted_source or "", predicted_target or "")
        prediction = predicted.get(key)
        if prediction is None:
            structural[relation_type][2] += 1
            predicted_start: set[str] = set()
            predicted_end: set[str] = set()
        else:
            structural[relation_type][0] += 1
            found.add(key)
            predicted_start = {min(prediction[0])} if prediction[0] else set()
            predicted_end = {min(prediction[1])} if prediction[1] else set()
        _date_counts(
            temporal[relation_type], {start} if start else set(), predicted_start
        )
        _date_counts(temporal[relation_type], {end} if end else set(), predicted_end)
    for key, (start, end) in predicted.items():
        if key in found:
            continue
        structural[key[0]][1] += 1
        temporal[key[0]][1] += int(bool(start)) + int(bool(end))
    return _metrics_from_counts(structural), _metrics_from_counts(temporal)


def _date_counts(counts: list[int], expected: set[str], predicted: set[str]) -> None:
    counts[0] += len(expected & predicted)
    counts[1] += len(predicted - expected)
    counts[2] += len(expected - predicted)


def _metrics_from_counts(counts: dict[str, list[int]]) -> MetricsByType:
    by_type = {
        name: metrics(values[0], values[1], values[2])
        for name, values in sorted(counts.items())
    }
    if not by_type:
        return MetricsByType({}, metrics(0, 0, 0), 0.0, 0.0, 0.0)
    count = len(by_type)
    return MetricsByType(
        by_type=by_type,
        aggregated=metrics(
            sum(value.true_positives for value in by_type.values()),
            sum(value.false_positives for value in by_type.values()),
            sum(value.false_negatives for value in by_type.values()),
        ),
        macro_precision=sum(value.precision for value in by_type.values()) / count,
        macro_recall=sum(value.recall for value in by_type.values()) / count,
        macro_f1=sum(value.f1 for value in by_type.values()) / count,
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > 10_000 or "\x00" in value:
        raise ValueError("Invalid extraction text")
    return value.strip()


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _valid_date(value: Any) -> str:
    text = _text(value)
    if text:
        date.fromisoformat(text)
    return text
