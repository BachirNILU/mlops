"""Evaluation metrics. Small, explicit implementations so you can read the formulas."""

from __future__ import annotations

import numpy as np


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of relevant documents found in the top k results."""
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result (0 if none). MRR = mean over queries."""
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def dedupe(seq: list[str]) -> list[str]:
    """Several chunks can come from one document; keep the first occurrence of each document."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two raters, corrected for agreement by chance. 1 = perfect, 0 = chance."""
    a, b = np.array(a, bool), np.array(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def confusion(pred: list[bool], truth: list[bool]) -> dict[str, int]:
    p, t = np.array(pred, bool), np.array(truth, bool)
    return {
        "tp": int((p & t).sum()),
        "fp": int((p & ~t).sum()),
        "fn": int((~p & t).sum()),
        "tn": int((~p & ~t).sum()),
    }


def expected_calibration_error(
    conf: list[float], correct: list[bool], bins: int = 5
) -> tuple[float, list[dict]]:
    """ECE: weighted mean gap between stated confidence and observed accuracy, per confidence bin."""
    conf_a, corr = np.array(conf, float), np.array(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (conf_a > lo) & (conf_a <= hi) if lo > 0 else (conf_a >= lo) & (conf_a <= hi)
        if m.sum() == 0:
            continue
        gap = abs(conf_a[m].mean() - corr[m].mean())
        ece += m.mean() * gap
        table.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(m.sum()),
                "mean_conf": round(float(conf_a[m].mean()), 2),
                "accuracy": round(float(corr[m].mean()), 2),
            }
        )
    return float(ece), table


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of a mean. With 20 questions, intervals are wide: report them."""
    rng = np.random.default_rng(seed)
    v = np.array(values, float)
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
