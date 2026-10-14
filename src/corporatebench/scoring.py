from __future__ import annotations

import re
from datetime import date
from typing import Any

ANSWER_TYPES = frozenset({"bool", "int", "date", "List[str]", "str"})


def score_answer(expected: Any, answer: str, answer_type: str) -> float:
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"Unsupported answer type: {answer_type}")

    try:
        parsed = parse_answer(answer, answer_type)
    except ValueError:
        return 0.0

    if answer_type == "List[str]":
        return _set_f1(expected, parsed)
    if answer_type == "str" or answer_type == "date":
        return float(str(expected).strip().casefold() == str(parsed).strip().casefold())
    return float(expected == parsed)


def parse_answer(answer: str, answer_type: str) -> str | int | bool | list[str]:
    if not isinstance(answer, str) or len(answer) > 1_000_000 or "\x00" in answer:
        raise ValueError("Invalid model output")
    value = answer.strip()

    if answer_type == "bool":
        normalized = value.casefold()
        if "true" in normalized or "yes" in normalized:
            return True
        if "false" in normalized or "no" in normalized:
            return False
        raise ValueError("Expected a boolean")

    if answer_type == "int":
        match = re.search(r"\d+", value)
        if match is None:
            raise ValueError("Expected an integer")
        return int(match.group())

    if answer_type == "date":
        match = re.search(r"\d{4}-\d{2}-\d{2}", value)
        if match is None:
            raise ValueError("Expected an ISO date")
        date.fromisoformat(match.group())
        return match.group()

    if answer_type == "List[str]":
        if value.casefold() in {"", "none", "null", "[]"}:
            return []
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].strip()
        if "," in value:
            return [
                _strip_quotes(item.strip())
                for item in value.split(",")
                if item.strip()
            ]
        quoted = re.findall(r'["\']([^"\']*)["\']', value)
        if quoted:
            return [item.strip() for item in quoted if item.strip()]
        lines = []
        for line in value.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "Based on")):
                cleaned = re.sub(r"^[\d.\-*+]+\s*", "", line).strip()
                if cleaned:
                    lines.append(cleaned)
        return lines or ([value] if value else [])

    if answer_type == "str":
        return value

    raise ValueError(f"Unsupported answer type: {answer_type}")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def _set_f1(expected: list[str], answer: list[str]) -> float:
    expected_set = {item.strip().casefold() for item in expected}
    answer_set = {item.strip().casefold() for item in answer}
    if not expected_set and not answer_set:
        return 1.0
    if not expected_set or not answer_set:
        return 0.0
    overlap = len(expected_set & answer_set)
    precision = overlap / len(answer_set)
    recall = overlap / len(expected_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
