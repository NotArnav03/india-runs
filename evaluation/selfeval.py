"""Local self-evaluation against the JD-grounded judgment set (gap item F).

The competition's ground truth is hidden, so we can't compute the official
score locally.  Instead we measure the ranker on the behavioral judgment set
(:mod:`evaluation.judgment_archetypes`) using the *same* composite formula the
organizers use (``submission_spec.docx`` §4):

    composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10

Relevance follows the spec: P@10 counts tier ≥ 3 as relevant; NDCG uses the
graded tiers with the canonical exponential gain; MAP treats any tier ≥ 1 as
relevant (precision across all relevance levels).  We also report the
**honeypot rate** in the head — the Stage-3 disqualifier.

All metric primitives are the vetted FAIMR implementations in
:mod:`evaluation.metrics`.  This module also offers an offline coordinate-
ascent weight tuner; it is a *diagnostic tool*, not wired into the shipped
weights (which are kept hand-set and explainable for the Stage-5 interview).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.judgment_archetypes import build_judgment_set, REF
from evaluation.metrics import ndcg_at_k, precision_at_k, average_precision
from ir.adapters import candidate_from_raw
from ir.honeypot import consistency_report


@dataclass
class EvalReport:
    composite: float
    ndcg10: float
    ndcg50: float
    map_: float
    p10: float
    honeypot_rate_top10: float
    n: int

    def pretty(self) -> str:
        return (
            f"Self-eval on {self.n} JD-grounded archetypes\n"
            f"  NDCG@10 : {self.ndcg10:.4f}  (weight 0.50)\n"
            f"  NDCG@50 : {self.ndcg50:.4f}  (weight 0.30)\n"
            f"  MAP     : {self.map_:.4f}  (weight 0.15)\n"
            f"  P@10    : {self.p10:.4f}  (weight 0.05)\n"
            f"  -------------------------------------------\n"
            f"  COMPOSITE: {self.composite:.4f}\n"
            f"  honeypot rate in top 10: {self.honeypot_rate_top10:.2%}"
        )


def composite_from_labels(y_true: list[int], y_scores: list[float]) -> EvalReport:
    """Compute the official composite from graded tiers + predicted scores."""
    rel3 = [1 if t >= 3 else 0 for t in y_true]   # P@10 relevance (tier 3+)
    rel1 = [1 if t >= 1 else 0 for t in y_true]   # MAP relevance (any level)
    ndcg10 = ndcg_at_k(y_true, y_scores, 10, gain="exponential")
    ndcg50 = ndcg_at_k(y_true, y_scores, 50, gain="exponential")
    p10 = precision_at_k(rel3, y_scores, 10)
    map_ = average_precision(rel1, y_scores)
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_ + 0.05 * p10
    return EvalReport(composite, ndcg10, ndcg50, map_, p10, 0.0, len(y_true))


def evaluate(
    embedder_name: str = "hashing",
    weights: dict[str, float] | None = None,
    reference_date: dt.date = REF,
    use_cross_encoder: bool = False,
) -> EvalReport:
    """Score the judgment set and return the composite + honeypot rate."""
    from rank import score_candidates  # imported here to avoid a cycle at module load

    js = build_judgment_set()
    golds = {raw["candidate_id"]: tier for raw, tier in js}
    candidates = [candidate_from_raw(raw) for raw, _ in js]
    raw_by_id = {raw["candidate_id"]: raw for raw, _ in js}

    final, _ = score_candidates(
        candidates, embedder_name=embedder_name, weights=weights,
        reference_date=reference_date, use_cross_encoder=use_cross_encoder,
    )

    ids = list(golds)
    y_true = [golds[i] for i in ids]
    y_scores = [final[i] for i in ids]
    report = composite_from_labels(y_true, y_scores)

    # honeypot rate in the predicted top-10.
    top10 = sorted(final, key=lambda i: (-final[i], i))[:10]
    suspects = sum(1 for i in top10 if consistency_report(raw_by_id[i], reference_date).is_suspect)
    report.honeypot_rate_top10 = suspects / max(1, len(top10))
    return report


def tune_weights(
    base_weights: dict[str, float],
    embedder_name: str = "hashing",
    rounds: int = 2,
    step: float = 0.04,
) -> tuple[dict[str, float], float]:
    """Offline coordinate-ascent on the blend weights (diagnostic only).

    Returns ``(best_weights, best_composite)``.  Not auto-applied to the
    shipped weights — used to confirm the hand-set weights are near a local
    optimum on the judgment set, and to surface obviously-wrong weightings.
    """
    weights = dict(base_weights)
    best = evaluate(embedder_name=embedder_name, weights=weights).composite
    for _ in range(rounds):
        for name in list(weights):
            for delta in (step, -step):
                trial = dict(weights)
                trial[name] = max(0.0, trial[name] + delta)
                score = evaluate(embedder_name=embedder_name, weights=trial).composite
                if score > best + 1e-9:
                    weights, best = trial, score
    return weights, best


if __name__ == "__main__":
    print(evaluate(embedder_name="hashing").pretty())
