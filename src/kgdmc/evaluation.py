from __future__ import annotations

import math


def precision_recall_at_percent(scores: list[float], labels: list[int], percent: float) -> dict[str, float | int]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and have equal length")
    if not 0 < percent <= 1:
        raise ValueError("percent must be in (0, 1]")
    k = max(1, math.ceil(len(scores) * percent))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = sum(1 for i in ranked[:k] if labels[i] == 1)
    positives = sum(labels)
    return {"k": k, "precision": hits / k, "recall": hits / max(1, positives)}


def evaluate_rankings(scores: list[float], labels: list[int]) -> dict[str, dict[str, float | int]]:
    return {f"{int(p * 100)}%": precision_recall_at_percent(scores, labels, p) for p in (0.01, 0.02, 0.03, 0.04, 0.05)}

