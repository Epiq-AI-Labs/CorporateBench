from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

JsonObject = dict[str, Any]
ToolHandler = Callable[[JsonObject], Awaitable[str]]


class ToolCallLimitExceeded(RuntimeError):
    """Raised when a benchmark question exceeds its tool-call budget."""


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_schema: JsonObject
    handler: ToolHandler = field(repr=False)

    async def call(self, arguments: JsonObject) -> str:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        return await self.handler(arguments)


@dataclass(frozen=True, slots=True)
class ModelRun:
    answer: str
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.answer, str)
            or len(self.answer) > 1_000_000
            or "\x00" in self.answer
        ):
            raise ValueError("Model answer is invalid")
        for name in ("tool_calls", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Model {name} is invalid")


class ModelBackend(Protocol):
    name: str

    async def run(
        self,
        *,
        system: str,
        prompt: str,
        tools: Sequence[Tool],
        max_tool_calls: int,
    ) -> ModelRun: ...


@dataclass(frozen=True, slots=True)
class QuestionResult:
    id: int
    task: str
    question: str
    answer_type: str
    expected: str | int | bool | list[str]
    answer: str
    score: float
    elapsed_seconds: float
    tool_calls: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    company: str
    method: str
    model: str
    questions: tuple[QuestionResult, ...]

    @property
    def score(self) -> float:
        if not self.questions:
            return 0.0
        return sum(question.score for question in self.questions) / len(self.questions)

    @property
    def scores_by_type(self) -> dict[str, float]:
        groups: dict[str, list[float]] = {}
        for question in self.questions:
            groups.setdefault(question.answer_type, []).append(question.score)
        return {
            answer_type: sum(scores) / len(scores)
            for answer_type, scores in groups.items()
        }

    @property
    def scores_by_task(self) -> dict[str, float]:
        groups: dict[str, list[float]] = {}
        for question in self.questions:
            groups.setdefault(question.task, []).append(question.score)
        return {task: sum(scores) / len(scores) for task, scores in groups.items()}

    def to_dict(self) -> JsonObject:
        return {
            "company": self.company,
            "method": self.method,
            "model": self.model,
            "score": self.score,
            "scores_by_task": self.scores_by_task,
            "scores_by_type": self.scores_by_type,
            "questions": [asdict(question) for question in self.questions],
        }
