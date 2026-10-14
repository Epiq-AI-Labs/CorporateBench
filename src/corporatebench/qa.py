from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import question_prompt, system_prompt
from .scoring import score_answer


@dataclass(frozen=True, slots=True)
class QATaskDefinition:
    name: str
    label: str

    def system_prompt(self, method: str, *, tool_limit: int) -> str:
        return system_prompt(method, self.name, tool_limit=tool_limit)

    def question_prompt(self, method: str, question: dict[str, Any]) -> str:
        return question_prompt(method, self.name, question)

    def score_answer(self, expected: Any, answer: str, answer_type: str) -> float:
        return score_answer(expected, answer, answer_type)
