from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Metrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricsByType:
    by_type: dict[str, Metrics]
    aggregated: Metrics
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_type": {
                name: metrics.to_dict() for name, metrics in self.by_type.items()
            },
            "aggregated": self.aggregated.to_dict(),
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


def metrics_by_type(
    expected: set[tuple[str, ...]], predicted: set[tuple[str, ...]]
) -> MetricsByType:
    names = sorted({item[0] for item in expected | predicted})
    by_type: dict[str, Metrics] = {}
    for name in names:
        expected_type = {item for item in expected if item[0] == name}
        predicted_type = {item for item in predicted if item[0] == name}
        true_positives = len(expected_type & predicted_type)
        by_type[name] = metrics(
            true_positives,
            len(predicted_type) - true_positives,
            len(expected_type) - true_positives,
        )
    if not by_type:
        return MetricsByType({}, metrics(0, 0, 0), 0.0, 0.0, 0.0)
    count = len(by_type)
    true_positives = sum(value.true_positives for value in by_type.values())
    false_positives = sum(value.false_positives for value in by_type.values())
    false_negatives = sum(value.false_negatives for value in by_type.values())
    return MetricsByType(
        by_type=by_type,
        aggregated=metrics(true_positives, false_positives, false_negatives),
        macro_precision=sum(value.precision for value in by_type.values()) / count,
        macro_recall=sum(value.recall for value in by_type.values()) / count,
        macro_f1=sum(value.f1 for value in by_type.values()) / count,
    )


def metrics(true_positives: int, false_positives: int, false_negatives: int) -> Metrics:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )
