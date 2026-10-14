from __future__ import annotations

import asyncio
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pyoxigraph import Literal, NamedNode
from sklearn.model_selection import train_test_split

from ..kb import KnowledgeBase
from ..metrics import MetricsByType, metrics
from ..types import ModelBackend, ModelRun

TASK_NAME = "topic_classification"
TASK_LABEL = "Topic Classification"

_CORE = "https://storybeat.company/ontology/core#"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_DOCUMENT = f"{_CORE}Document"
_LEVEL = {"biocure": "firstParentTopic", "pound": "secondParentTopic"}
_ROOT = {"biocure": "thirdParentTopic", "pound": "fourthParentTopic"}
_TRAINING_SIZES = {
    "biocure": (0, 10, 50, 100, 1_000, 10_000),
    "pound": (0, 10, 50, 100, 1_000, 10_000, 100_000),
}
_TOPIC_COUNTS = {"biocure": 31, "pound": 17}
_MAX_DOCUMENT_CHARS = 100_000
_MAX_FEW_SHOT_CHARS = 2_000_000

_PROMPT = """You are an expert topic classification system.

INSTRUCTIONS:
1. Return ONLY one of the provided labels, do not generate extra unnecessary text.
2. Do NOT generate an empty text as an answer, you ALWAYS MUST return one of the provided labels

{few_shot_examples}

# Your task:
Classify the following document with one the labels provided (DO NOT REASON):

## DOCUMENT TO ANALYZE:

{document_content}

## LABELS:

{topics}

## RESPONSE:
"""


@dataclass(frozen=True, slots=True)
class TopicPrediction:
    filename: str
    expected: str
    predicted: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TopicClassificationRun:
    seed: int
    shots: int
    metrics: MetricsByType
    predictions: tuple[TopicPrediction, ...]

    @property
    def score(self) -> float:
        return self.metrics.macro_f1

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "shots": self.shots,
            "score": self.score,
            "metrics": self.metrics.to_dict(),
            "predictions": [asdict(prediction) for prediction in self.predictions],
        }


@dataclass(frozen=True, slots=True)
class TopicClassificationResult:
    company: str
    model: str
    runs: tuple[TopicClassificationRun, ...]

    @property
    def scores_by_shots(self) -> dict[int, float]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for run in self.runs:
            grouped[run.shots].append(run.score)
        return {shots: sum(scores) / len(scores) for shots, scores in grouped.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "model": self.model,
            "scores_by_shots": self.scores_by_shots,
            "runs": [run.to_dict() for run in self.runs],
        }


async def run_topic_classification(
    kb: KnowledgeBase,
    *,
    company: str,
    model: ModelBackend,
    shots: tuple[int, ...],
    repetitions: int,
    test_size: int,
    concurrency: int,
) -> TopicClassificationResult:
    documents = await asyncio.to_thread(_load_documents, kb, company)
    labels = tuple(sorted({document[1] for document in documents}))
    if len(labels) != _TOPIC_COUNTS[company]:
        raise ValueError("KB topic taxonomy does not match CorporateBench")

    content_indices = {name: index for index, name in enumerate(kb.document_names)}
    semaphore = asyncio.Semaphore(concurrency)
    runs: list[TopicClassificationRun] = []
    for seed in range(repetitions):
        splits = await asyncio.to_thread(
            _paper_splits, documents, company, seed, test_size
        )
        for shot_count in shots:
            train, test = splits[shot_count]
            few_shot = _few_shot_examples(kb, content_indices, train, labels)

            async def classify(
                document: tuple[str, str, str],
                few_shot_examples: str = few_shot,
            ) -> TopicPrediction:
                filename, expected, _ = document
                try:
                    async with semaphore:
                        _, content = kb.read_document(content_indices[filename])
                        content = _validate_document_content(content)
                        prompt = _classification_prompt(
                            content, labels, few_shot_examples
                        )
                        response = await model.run(
                            system="",
                            prompt=prompt,
                            tools=(),
                            max_tool_calls=0,
                        )
                    if not isinstance(response, ModelRun):
                        raise TypeError("model backend returned an invalid result")
                    predicted = _clean_prediction(response.answer)
                    error = None if predicted in labels else "invalid_label"
                except Exception:
                    predicted = ""
                    error = "model_error"
                return TopicPrediction(filename, expected, predicted, error)

            predictions = tuple(
                await asyncio.gather(*(classify(item) for item in test))
            )
            runs.append(
                TopicClassificationRun(
                    seed=seed,
                    shots=shot_count,
                    metrics=_classification_metrics(predictions, labels),
                    predictions=predictions,
                )
            )
    return TopicClassificationResult(company, model.name, tuple(runs))


def _load_documents(kb: KnowledgeBase, company: str) -> list[tuple[str, str, str]]:
    if company not in _LEVEL:
        raise ValueError("Topic Classification is defined for biocure and pound")
    wanted = {
        "filename",
        "generationType",
        "topic",
        _LEVEL[company],
        _ROOT[company],
    }
    names = set(kb.document_names)
    documents: list[tuple[str, str, str]] = []
    with TemporaryDirectory(prefix="corporatebench-topics-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "metadata.sqlite3")
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute(
                "CREATE TABLE typed (subject TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE properties ("
                "subject TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY (subject, name)) WITHOUT ROWID"
            )
            typed_rows: list[tuple[str]] = []
            property_rows: list[tuple[str, str, str]] = []
            for quad in kb.iter_graph_quads():
                if not isinstance(quad.subject, NamedNode):
                    continue
                subject = quad.subject.value
                if not 1 <= len(subject) <= 10_000 or "\x00" in subject:
                    raise ValueError("KB contains invalid topic metadata")
                if (
                    quad.predicate.value == _RDF_TYPE
                    and isinstance(quad.object, NamedNode)
                    and quad.object.value == _DOCUMENT
                ):
                    typed_rows.append((subject,))
                elif isinstance(quad.object, Literal):
                    name = quad.predicate.value.rsplit("#", 1)[-1]
                    value = quad.object.value
                    if name in wanted:
                        if len(value) > 10_000 or "\x00" in value:
                            raise ValueError("KB contains invalid topic metadata")
                        property_rows.append((subject, name, value))
                if len(typed_rows) + len(property_rows) >= 10_000:
                    _write_topic_metadata(connection, typed_rows, property_rows)
            _write_topic_metadata(connection, typed_rows, property_rows)

            rows = connection.execute(
                "SELECT filename.value, generation.value, level.value, "
                "root.value, topic.value "
                "FROM typed "
                "JOIN properties AS filename ON filename.subject = typed.subject "
                "AND filename.name = 'filename' "
                "JOIN properties AS generation ON generation.subject = typed.subject "
                "AND generation.name = 'generationType' "
                "JOIN properties AS level ON level.subject = typed.subject "
                "AND level.name = ? "
                "LEFT JOIN properties AS root ON root.subject = typed.subject "
                "AND root.name = ? "
                "JOIN properties AS topic ON topic.subject = typed.subject "
                "AND topic.name = 'topic' "
                "ORDER BY typed.subject",
                (_LEVEL[company], _ROOT[company]),
            )
            for filename, generation, level, root, topic in rows:
                label = "Non-business" if root == "Watercooler Talk" else level
                if (
                    generation.rsplit(".", 1)[-1].casefold() == "relationship"
                    and filename in names
                    and label
                    and topic
                ):
                    documents.append((filename, label, topic))
        finally:
            connection.close()
    if not documents:
        raise ValueError("KB contains no Topic Classification documents")
    return documents


def _write_topic_metadata(
    connection: sqlite3.Connection,
    typed_rows: list[tuple[str]],
    property_rows: list[tuple[str, str, str]],
) -> None:
    if typed_rows:
        connection.executemany(
            "INSERT OR IGNORE INTO typed (subject) VALUES (?)", typed_rows
        )
        typed_rows.clear()
    if property_rows:
        connection.executemany(
            "INSERT INTO properties (subject, name, value) VALUES (?, ?, ?) "
            "ON CONFLICT (subject, name) DO UPDATE SET value = excluded.value",
            property_rows,
        )
        property_rows.clear()


def _paper_splits(
    documents: list[tuple[str, str, str]],
    company: str,
    seed: int,
    test_size: int,
) -> dict[int, tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]]:
    previous_train: list[tuple[str, str, str]] | None = None
    previous_test: list[tuple[str, str, str]] | None = None
    splits: dict[
        int, tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]
    ] = {}
    for train_size in reversed(_TRAINING_SIZES[company]):
        if previous_train is None:
            effective_train = min(train_size, len(documents) - test_size)
            if effective_train < 1:
                raise ValueError("Not enough documents for the paper split")
            train, test = train_test_split(
                documents,
                train_size=effective_train,
                test_size=test_size,
                random_state=seed,
                stratify=[document[1] for document in documents],
            )
        elif train_size == 0:
            train, test = [], previous_test
        elif train_size >= len(previous_train):
            train, test = previous_train, previous_test
        else:
            counts = Counter(document[1] for document in previous_train)
            stratify = (
                [document[1] for document in previous_train]
                if min(counts.values()) >= 2
                else None
            )
            train, _ = train_test_split(
                previous_train,
                train_size=train_size,
                random_state=seed,
                stratify=stratify,
            )
            test = previous_test
        splits[train_size] = (list(train), list(test))
        previous_train, previous_test = list(train), list(test)
    return splits


def _few_shot_examples(
    kb: KnowledgeBase,
    indices: dict[str, int],
    documents: list[tuple[str, str, str]],
    labels: tuple[str, ...],
) -> str:
    if not documents:
        return "# Few-shot examples:\n\n\n" + "-" * 50 + "\n\n"
    topics = "".join(f"- {label}\n" for label in labels)
    examples: list[str] = []
    for filename, label, _ in documents:
        _, content = kb.read_document(indices[filename])
        content = _validate_document_content(content)
        examples.append(
            "## DOCUMENT TO ANALYZE:\n\n"
            f"{content}\n\n## LABELS:\n\n{topics}\n## RESPONSE:\n\n{label}\n"
        )
    separator = "\n" + "-" * 50 + "\n"
    result = "# Few-shot examples:\n\n" + separator.join(examples) + separator + "\n"
    if len(result) > _MAX_FEW_SHOT_CHARS:
        raise ValueError("Few-shot prompt exceeds the benchmark size limit")
    return result


def _classification_prompt(
    content: str, labels: tuple[str, ...], few_shot_examples: str
) -> str:
    topics = "".join(f"- {label}\n" for label in labels)
    return _PROMPT.format(
        few_shot_examples=few_shot_examples,
        document_content=content,
        topics=topics,
    )


def _validate_document_content(content: str) -> str:
    if not isinstance(content, str) or len(content) > _MAX_DOCUMENT_CHARS:
        raise ValueError("Classification document exceeds the benchmark size limit")
    return content


def _clean_prediction(prediction: str) -> str:
    marker = "RESPONSE:"
    if marker in prediction:
        prediction = prediction[prediction.find(marker) + len(marker) :]
    return prediction.strip()


def _classification_metrics(
    predictions: tuple[TopicPrediction, ...], labels: tuple[str, ...]
) -> MetricsByType:
    by_type = {}
    for label in labels:
        true_positives = sum(
            prediction.expected == label and prediction.predicted == label
            for prediction in predictions
        )
        false_positives = sum(
            prediction.expected != label and prediction.predicted == label
            for prediction in predictions
        )
        false_negatives = sum(
            prediction.expected == label and prediction.predicted != label
            for prediction in predictions
        )
        by_type[label] = metrics(true_positives, false_positives, false_negatives)
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
