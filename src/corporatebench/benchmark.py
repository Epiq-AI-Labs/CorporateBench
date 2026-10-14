from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Iterable
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from .kb import KnowledgeBase
from .scoring import ANSWER_TYPES
from .sql import SqlIndexInfo, SqlKnowledgeBase
from .tasks.integrated_qa import TASK as _INTEGRATED_QA
from .tasks.kb_evaluation import TASK_NAME as _KB_EVALUATION_NAME
from .tasks.kb_evaluation import KBEvaluationResult
from .tasks.kb_evaluation import run_kb_evaluation as _run_kb_evaluation
from .tasks.kb_qa import TASK as _KB_QA
from .tasks.topic_classification import TASK_NAME as _TOPIC_CLASSIFICATION_NAME
from .tasks.topic_classification import TopicClassificationResult
from .tasks.topic_classification import (
    run_topic_classification as _run_topic_classification,
)
from .tasks.topic_qa import TASK as _TOPIC_QA
from .types import (
    BenchmarkResult,
    ModelBackend,
    ModelRun,
    QuestionResult,
    Tool,
    ToolCallLimitExceeded,
)
from .vectors import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    EmbeddingBackend,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    VectorIndex,
    VectorIndexInfo,
    VectorProgress,
)

Company = Literal["zenith", "streamvibe", "biocure", "pound"]
Method = Literal["rag", "kb"]
QATask = Literal["kb_qa", "topic_qa", "integrated_qa"]
Task = Literal[
    "kb_evaluation",
    "topic_classification",
    "kb_qa",
    "topic_qa",
    "integrated_qa",
]

COMPANIES = frozenset({"zenith", "streamvibe", "biocure", "pound"})
_QA_TASK_BY_NAME = {
    task.name: task for task in (_KB_QA, _TOPIC_QA, _INTEGRATED_QA)
}
QA_TASKS = tuple(_QA_TASK_BY_NAME)
TASKS = (_KB_EVALUATION_NAME, _TOPIC_CLASSIFICATION_NAME, *QA_TASKS)


class CorporateBench:
    tasks = TASKS
    qa_tasks = QA_TASKS

    def __init__(
        self,
        data_dir: str | os.PathLike[str] | None = None,
        *,
        company: Company = "zenith",
        top_k: int = 10,
        tool_call_limit: int = 5,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_revision: str | None = None,
        embedding_batch_size: int = 32,
        embedding_device: str | None = None,
        embedder: EmbeddingBackend | None = None,
        vector_cache_dir: str | os.PathLike[str] | None = None,
        sql_cache_dir: str | os.PathLike[str] | None = None,
        paper_embeddings: bool = False,
    ) -> None:
        if not isinstance(paper_embeddings, bool):
            raise TypeError("paper_embeddings must be a boolean")
        if company not in COMPANIES:
            raise ValueError(f"Unknown company: {company}")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 50
        ):
            raise ValueError("top_k must be between 1 and 50")
        if (
            isinstance(tool_call_limit, bool)
            or not isinstance(tool_call_limit, int)
            or not 1 <= tool_call_limit <= 20
        ):
            raise ValueError("tool_call_limit must be between 1 and 20")

        if data_dir is None:
            data_dir = _download_company(company)
        self.data_dir = Path(data_dir).resolve(strict=True)
        if not self.data_dir.is_dir():
            raise ValueError("data_dir must be a directory")
        self.company = company
        self.top_k = top_k
        self.tool_call_limit = tool_call_limit

        kb_path = self._safe_data_file(Path("kb") / f"{company}.kb")
        self.kb = KnowledgeBase(kb_path)
        try:
            if embedder is None:
                if paper_embeddings:
                    if (
                        embedding_model != DEFAULT_EMBEDDING_MODEL
                        or embedding_revision is not None
                        or embedding_device is not None
                    ):
                        raise ValueError(
                            "paper_embeddings cannot be combined with local "
                            "model options"
                        )
                    embedder = OpenAIEmbedder()
                else:
                    revision = (
                        DEFAULT_EMBEDDING_REVISION
                        if embedding_model == DEFAULT_EMBEDDING_MODEL
                        and embedding_revision is None
                        else embedding_revision
                    )
                    embedder = SentenceTransformerEmbedder(
                        embedding_model,
                        revision=revision,
                        batch_size=embedding_batch_size,
                        device=embedding_device,
                    )
            elif paper_embeddings:
                raise ValueError("paper_embeddings cannot be combined with embedder")
            self.vectors = VectorIndex(
                self.kb,
                company=company,
                embedder=embedder,
                cache_dir=vector_cache_dir,
                batch_size=embedding_batch_size,
            )
            self.sql = SqlKnowledgeBase(
                self.kb,
                company=company,
                cache_dir=sql_cache_dir,
            )
        except Exception:
            self.kb.close()
            raise

    async def __aenter__(self) -> CorporateBench:
        return self

    async def __aexit__(self, *_: object) -> None:
        self.close()

    def __enter__(self) -> CorporateBench:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.kb.close()

    async def bootstrap_vectors(
        self,
        progress: Callable[[VectorProgress], None] | None = None,
    ) -> VectorIndexInfo:
        return await asyncio.to_thread(self.vectors.bootstrap, progress)

    def bootstrap_vectors_sync(
        self, progress: Callable[[VectorProgress], None] | None = None
    ) -> VectorIndexInfo:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.bootstrap_vectors(progress))
        raise RuntimeError(
            "Use 'await benchmark.bootstrap_vectors()' inside an event loop"
        )

    async def bootstrap_sql(self) -> SqlIndexInfo:
        return await asyncio.to_thread(self.sql.bootstrap)

    def bootstrap_sql_sync(self) -> SqlIndexInfo:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.bootstrap_sql())
        raise RuntimeError("Use 'await benchmark.bootstrap_sql()' inside an event loop")

    async def run(
        self,
        model: ModelBackend,
        *,
        method: Method = "rag",
        tasks: Iterable[QATask] = QA_TASKS,
        limit: int | None = None,
    ) -> BenchmarkResult:
        if method not in {"rag", "kb"}:
            raise ValueError("method must be 'rag' or 'kb'")
        selected_tasks = tuple(tasks)
        if not selected_tasks or any(task not in QA_TASKS for task in selected_tasks):
            raise ValueError("Unknown QA task")
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 10_000
        ):
            raise ValueError("limit must be between 1 and 10000")
        _validate_model(model)

        if method == "rag":
            await self.bootstrap_vectors()
        else:
            await self.bootstrap_sql()

        results: list[QuestionResult] = []
        for task in selected_tasks:
            task_definition = _QA_TASK_BY_NAME[task]
            system = task_definition.system_prompt(
                method, tool_limit=self.tool_call_limit
            )
            questions = self._load_questions(task)
            if limit is not None:
                questions = questions[:limit]
            for question in questions:
                tools, tool_count = _limited_tools(
                    self._tools(method, task), self.tool_call_limit
                )
                started = time.monotonic()
                answer = ""
                tool_calls = 0
                error: str | None = None
                try:
                    run = await model.run(
                        system=system,
                        prompt=task_definition.question_prompt(method, question),
                        tools=tools,
                        max_tool_calls=self.tool_call_limit,
                    )
                    if not isinstance(run, ModelRun):
                        raise TypeError("model backend returned an invalid result")
                    tool_calls = max(run.tool_calls, tool_count())
                    if tool_calls > self.tool_call_limit:
                        raise ToolCallLimitExceeded(
                            f"Tool call limit of {self.tool_call_limit} exceeded"
                        )
                    answer = run.answer
                    score = task_definition.score_answer(
                        question["answer"], answer, question["answer_type"]
                    )
                except ToolCallLimitExceeded:
                    score = 0.0
                    tool_calls = max(tool_calls, tool_count())
                    error = "tool_call_limit_exceeded"
                except Exception:
                    score = 0.0
                    tool_calls = max(tool_calls, tool_count())
                    error = "model_error"
                results.append(
                    QuestionResult(
                        id=question["id"],
                        task=task,
                        question=question["question"],
                        answer_type=question["answer_type"],
                        expected=question["answer"],
                        answer=answer,
                        score=score,
                        elapsed_seconds=time.monotonic() - started,
                        tool_calls=tool_calls,
                        error=error,
                    )
                )

        return BenchmarkResult(
            company=self.company,
            method=method,
            model=model.name,
            questions=tuple(results),
        )

    def run_sync(self, model: ModelBackend, **kwargs: Any) -> BenchmarkResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(model, **kwargs))
        raise RuntimeError("Use 'await benchmark.run(...)' inside an event loop")

    async def run_kb_evaluation(
        self,
        model: ModelBackend,
        *,
        max_requests: int | None = None,
        concurrency: int = 8,
    ) -> KBEvaluationResult:
        _validate_model(model)
        _require_request_budget(max_requests, self.kb_evaluation_requests)
        _validate_concurrency(concurrency)
        return await _run_kb_evaluation(
            self.kb,
            company=self.company,
            model=model,
            concurrency=concurrency,
        )

    def run_kb_evaluation_sync(
        self, model: ModelBackend, **kwargs: Any
    ) -> KBEvaluationResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_kb_evaluation(model, **kwargs))
        raise RuntimeError(
            "Use 'await benchmark.run_kb_evaluation(...)' inside an event loop"
        )

    @property
    def kb_evaluation_requests(self) -> int:
        return sum(
            not name.casefold().startswith("calendar")
            for name in self.kb.document_names
        )

    async def run_topic_classification(
        self,
        model: ModelBackend,
        *,
        shots: Iterable[int] = (0, 10, 50),
        repetitions: int = 5,
        test_size: int = 1_000,
        max_requests: int | None = None,
        concurrency: int = 8,
    ) -> TopicClassificationResult:
        _validate_model(model)
        _validate_topic_classification_model(model)
        selected_shots = tuple(islice(iter(shots), 4))
        if (
            not selected_shots
            or any(
                isinstance(shot, bool)
                or not isinstance(shot, int)
                or shot not in {0, 10, 50}
                for shot in selected_shots
            )
            or len(selected_shots) != len(set(selected_shots))
        ):
            raise ValueError("shots must be unique values from 0, 10, and 50")
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or not 1 <= repetitions <= 5
        ):
            raise ValueError("repetitions must be between 1 and 5")
        if (
            isinstance(test_size, bool)
            or not isinstance(test_size, int)
            or not 1 <= test_size <= 1_000
        ):
            raise ValueError("test_size must be between 1 and 1000")
        minimum_test_size = {"biocure": 31, "pound": 17}.get(self.company)
        if minimum_test_size is None:
            raise ValueError("Topic Classification is defined for biocure and pound")
        if test_size < minimum_test_size:
            raise ValueError(
                f"test_size must be at least {minimum_test_size} for {self.company}"
            )
        _require_request_budget(
            max_requests, len(selected_shots) * repetitions * test_size
        )
        _validate_concurrency(concurrency)
        return await _run_topic_classification(
            self.kb,
            company=self.company,
            model=model,
            shots=selected_shots,
            repetitions=repetitions,
            test_size=test_size,
            concurrency=concurrency,
        )

    async def run_task(
        self,
        task: Task,
        model: ModelBackend,
        **kwargs: Any,
    ) -> KBEvaluationResult | TopicClassificationResult | BenchmarkResult:
        if task == _KB_EVALUATION_NAME:
            return await self.run_kb_evaluation(model, **kwargs)
        if task == _TOPIC_CLASSIFICATION_NAME:
            return await self.run_topic_classification(model, **kwargs)
        if task in QA_TASKS:
            return await self.run(model, tasks=(task,), **kwargs)
        raise ValueError(f"Unknown benchmark task: {task}")

    def run_task_sync(
        self,
        task: Task,
        model: ModelBackend,
        **kwargs: Any,
    ) -> KBEvaluationResult | TopicClassificationResult | BenchmarkResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_task(task, model, **kwargs))
        raise RuntimeError("Use 'await benchmark.run_task(...)' inside an event loop")

    def run_topic_classification_sync(
        self, model: ModelBackend, **kwargs: Any
    ) -> TopicClassificationResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_topic_classification(model, **kwargs))
        raise RuntimeError(
            "Use 'await benchmark.run_topic_classification(...)' inside an event loop"
        )

    def _tools(self, method: Method, task: QATask) -> tuple[Tool, ...]:
        if method == "rag":

            async def search(arguments: dict[str, Any]) -> str:
                if (
                    "search_query" not in arguments
                    or not set(arguments) <= {"search_query", "top_k"}
                    or not isinstance(arguments["search_query"], str)
                ):
                    raise ValueError("Expected a search_query string")
                requested = arguments.get("top_k")
                if requested is None:
                    requested = self.top_k
                if (
                    isinstance(requested, bool)
                    or not isinstance(requested, int)
                    or not 1 <= requested <= 50
                ):
                    raise ValueError("top_k must be between 1 and 50")
                return await asyncio.to_thread(
                    self.vectors.search,
                    arguments["search_query"],
                    limit=requested,
                )

            return (
                Tool(
                    name="retrieve",
                    description=(
                        "Retrieve documentation sections based on a search query."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "minLength": 2,
                                "maxLength": 1000,
                            },
                            "top_k": {
                                "type": ["integer", "null"],
                                "minimum": 1,
                                "maximum": 50,
                            },
                        },
                        "required": ["search_query"],
                        "additionalProperties": False,
                    },
                    handler=search,
                ),
            )

        async def execute_sql(arguments: dict[str, Any]) -> str:
            if set(arguments) != {"sql_query"} or not isinstance(
                arguments["sql_query"], str
            ):
                raise ValueError("Expected a sql_query string")
            return await asyncio.to_thread(
                self.sql.execute,
                arguments["sql_query"],
                task=task,
            )

        return (
            Tool(
                name="execute_sql",
                description="Execute a SQL query and return the results.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "minLength": 6,
                            "maxLength": SqlKnowledgeBase.MAX_QUERY_CHARS,
                        }
                    },
                    "required": ["sql_query"],
                    "additionalProperties": False,
                },
                handler=execute_sql,
            ),
        )

    def _load_questions(self, task: str) -> list[dict[str, Any]]:
        path = self._safe_data_file(Path(task) / f"{self.company}_questions.json")
        if path.stat().st_size > 50_000_000:
            raise ValueError("QA dataset is too large")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Invalid QA dataset")
        metadata = payload.get("metadata")
        questions = payload.get("questions")
        if (
            not isinstance(metadata, dict)
            or metadata.get("dataset_type") != task
            or metadata.get("kb") != f"{self.company}.kb"
            or not isinstance(questions, list)
            or not 1 <= len(questions) <= 10_000
        ):
            raise ValueError("Invalid QA dataset metadata")
        return [_validate_question(question) for question in questions]

    def _safe_data_file(self, relative: Path) -> Path:
        candidate = self.data_dir / relative
        if candidate.is_symlink():
            raise ValueError("Dataset files may not be symlinks")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(self.data_dir) or not resolved.is_file():
            raise ValueError("Dataset file is outside data_dir")
        return resolved


def _limited_tools(
    tools: tuple[Tool, ...], limit: int
) -> tuple[tuple[Tool, ...], Callable[[], int]]:
    used = 0

    def count() -> int:
        return used

    limited: list[Tool] = []
    for tool in tools:

        async def call(arguments: dict[str, Any], *, inner: Tool = tool) -> str:
            nonlocal used
            if used >= limit:
                raise ToolCallLimitExceeded(f"Tool call limit of {limit} exceeded")
            used += 1
            return await inner.call(arguments)

        limited.append(
            Tool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                handler=call,
            )
        )
    return tuple(limited), count


def _validate_model(model: ModelBackend) -> None:
    if (
        not isinstance(getattr(model, "name", None), str)
        or not 1 <= len(model.name) <= 200
        or any(character in model.name for character in "\x00\r\n")
        or not callable(getattr(model, "run", None))
    ):
        raise TypeError("model must implement ModelBackend")


def _validate_concurrency(concurrency: int) -> None:
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or not 1 <= concurrency <= 64
    ):
        raise ValueError("concurrency must be between 1 and 64")


def _require_request_budget(max_requests: int | None, planned_requests: int) -> None:
    if (
        isinstance(max_requests, bool)
        or not isinstance(max_requests, int)
        or not planned_requests <= max_requests <= 1_000_000
    ):
        raise ValueError(
            f"This run requires {planned_requests} model requests; pass "
            f"max_requests={planned_requests} or greater"
        )


def _validate_topic_classification_model(model: ModelBackend) -> None:
    output_limit = getattr(model, "max_output_tokens", None)
    if output_limit is None:
        output_limit = getattr(model, "max_tokens", None)
    if output_limit != 500:
        raise ValueError("Topic Classification requires a 500-token output limit")
    temperature = getattr(model, "temperature", None)
    if temperature is not None and temperature != 1:
        raise ValueError("Topic Classification requires temperature=1")


def _validate_question(question: Any) -> dict[str, Any]:
    if not isinstance(question, dict):
        raise ValueError("Invalid question")
    question_id = question.get("id")
    text = question.get("question")
    answer_type = question.get("answer_type")
    answer = question.get("answer")
    if not isinstance(question_id, int) or question_id < 0:
        raise ValueError("Invalid question id")
    if not isinstance(text, str) or not 1 <= len(text) <= 20_000 or "\x00" in text:
        raise ValueError("Invalid question text")
    if answer_type not in ANSWER_TYPES:
        raise ValueError("Invalid answer type")
    if not _valid_ground_truth(answer_type, answer):
        raise ValueError("Invalid ground-truth answer")
    return {
        "id": question_id,
        "question": text,
        "answer_type": answer_type,
        "answer": answer,
    }


def _valid_ground_truth(answer_type: str, answer: Any) -> bool:
    if answer_type == "bool":
        return isinstance(answer, bool)
    if answer_type == "int":
        return isinstance(answer, int) and not isinstance(answer, bool)
    if answer_type == "List[str]":
        return (
            isinstance(answer, list)
            and len(answer) <= 100_000
            and all(
                isinstance(item, str) and len(item) <= 10_000 and "\x00" not in item
                for item in answer
            )
        )
    if not isinstance(answer, str) or len(answer) > 20_000 or "\x00" in answer:
        return False
    if answer_type == "date":
        try:
            date.fromisoformat(answer)
        except ValueError:
            return False
    return True


def _download_company(company: str) -> Path:
    from huggingface_hub import snapshot_download

    destination = Path.home() / ".cache" / "corporatebench"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        # Public data: https://huggingface.co/datasets/epiq-ai-labs/corporatebench
        repo_id="epiq-ai-labs/corporatebench",
        repo_type="dataset",
        revision="d8fc7f3e22e4f6df3c7b4023cd680612d81e448f",
        token=False,
        local_dir=destination,
        allow_patterns=[
            f"data/kb/{company}.kb",
            *(f"data/{task}/{company}_questions.json" for task in QA_TASKS),
        ],
    )
    return destination / "data"
