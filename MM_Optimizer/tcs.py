"""
tcs.py  —  Trajectory Context Summarizer
-----------------------------------------
Pure deterministic Python (~80 lines, no LLM).
Converts a list of (round_idx, params, coverage, mean_time, violated?)
into a compact text block placed just before the LLM refinement ask.

Never scalarises to a single score — always reports the Pareto front.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import MM_Optimizer.search_config as SC


@dataclass
class RoundResult:
    round_idx:  int
    params:     dict
    coverage:   float           # 0–1
    mean_time:  float           # seconds
    violated:   bool = False    # True if constraint was rejected before evaluation


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def summarize(results: list[RoundResult], max_pareto_shown: int = SC.TCS_MAX_PARETO_SHOWN) -> str:
    """Return a compact TCS block ready for injection into the LLM prompt."""
    if not results:
        return "(no rounds completed yet)"

    valid = [r for r in results if not r.violated]
    n_total   = len(results)
    n_violated = len(results) - len(valid)

    pareto   = _pareto_front(valid)
    best_cov = max((r.coverage  for r in valid), default=0.0)
    best_time = min((r.mean_time for r in valid), default=float("inf"))

    lines = []
    lines.append(f"ROUND {n_total}/{n_total} -- objectives: maximize coverage (primary), minimize time (secondary)")
    lines.append("")

    # Pareto front
    lines.append(f"Pareto-best-so-far (non-dominated, up to {SC.TCS_MAX_PARETO_SHOWN} shown):")
    shown = pareto[:max_pareto_shown]
    for r in shown:
        param_str = _short_params(r.params)
        lines.append(f"  cov={r.coverage:.2f}  time={r.mean_time:.2f}s  @ {param_str}")
    if len(pareto) > max_pareto_shown:
        lines.append(f"  ... and {len(pareto) - max_pareto_shown} more non-dominated configs")
    lines.append("  (each is optimal under a different time budget -- do not average them)")
    lines.append("")

    # Per-parameter effect (requires >=2 valid rounds)
    effects = _parameter_effects(valid)
    if effects:
        lines.append("Per-parameter effect (sign of objective change when this param moved, others ~fixed):")
        for param, (cov_sign, time_sign, example) in effects.items():
            lines.append(f"  {param:<36s} -> coverage {cov_sign:<4s}  time {time_sign}  {example}")
        lines.append("")

    # Hard failures
    if n_violated > 0:
        violated_params = [r.params for r in results if r.violated]
        constraint_msgs = _constraint_violations(violated_params)
        lines.append(f"Hard failures to avoid ({n_violated} constraint violation(s)):")
        for msg in constraint_msgs:
            lines.append(f"  {msg}")
        lines.append("")

    # Frontier gaps
    gaps = _frontier_gaps(valid, best_cov, best_time)
    if gaps:
        lines.append("Frontier gaps (unexplored regions -- propose configs here):")
        for gap in gaps:
            lines.append(f"  {gap}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pareto front
# ─────────────────────────────────────────────────────────────────────────────

def _pareto_front(results: list[RoundResult]) -> list[RoundResult]:
    """Non-dominated set: maximise coverage, minimise time. Sorted by coverage desc."""
    front = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            # other dominates r if it's at least as good on both and strictly better on one
            if (other.coverage >= r.coverage and other.mean_time <= r.mean_time and
                    (other.coverage > r.coverage or other.mean_time < r.mean_time)):
                dominated = True
                break
        if not dominated:
            front.append(r)
    front.sort(key=lambda r: -r.coverage)
    return front


# ─────────────────────────────────────────────────────────────────────────────
# Per-parameter effect analysis
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_PARAMS = {
    "refStep", "distQuantification", "angleQuantification", "referredStep",
    "maxVoteRatio", "outputNum", "operationApproach", "deviationCorrectionCapacity",
    "confidenceThreshold", "scoreLevel",
}

def _parameter_effects(results: list[RoundResult]) -> dict[str, tuple[str, str, str]]:
    """Return {param: (cov_sign, time_sign, example_str)} for params that moved."""
    if len(results) < 2:
        return {}
    effects: dict[str, tuple[str, str, str]] = {}
    params_union = set()
    for r in results:
        params_union.update(r.params.keys())

    for param in sorted(params_union):
        if param not in _NUMERIC_PARAMS:
            continue
        # Find pairs where this param changed and others stayed roughly constant
        deltas_cov, deltas_time, example = [], [], ""
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                ri, rj = results[i], results[j]
                vi = ri.params.get(param)
                vj = rj.params.get(param)
                if vi is None or vj is None or vi == vj:
                    continue
                try:
                    delta_p = float(vj) - float(vi)
                except (TypeError, ValueError):
                    continue
                if delta_p == 0:
                    continue
                dc = rj.coverage  - ri.coverage
                dt = rj.mean_time - ri.mean_time
                deltas_cov.append(dc / delta_p)
                deltas_time.append(dt / delta_p)
                if not example:
                    example = (f"({vi}->{vj} gave {dc:+.2f} cov, {dt:+.2f}s)")
        if not deltas_cov:
            continue
        avg_dc = sum(deltas_cov) / len(deltas_cov)
        avg_dt = sum(deltas_time) / len(deltas_time)
        cov_sign  = "up" if avg_dc > SC.TCS_COV_SENSITIVITY  else ("dn" if avg_dc < -SC.TCS_COV_SENSITIVITY  else "~")
        time_sign = "up" if avg_dt > SC.TCS_TIME_SENSITIVITY else ("dn" if avg_dt < -SC.TCS_TIME_SENSITIVITY else "~")
        effects[param] = (cov_sign, time_sign, example)
    return effects


# ─────────────────────────────────────────────────────────────────────────────
# Constraint violation summary
# ─────────────────────────────────────────────────────────────────────────────

def _constraint_violations(violated_params: list[dict]) -> list[str]:
    msgs: dict[str, int] = {}
    for p in violated_params:
        ref = p.get("referredStep")
        rfk = p.get("refStep")
        if ref is not None and rfk is not None and ref > rfk:
            key = f"referredStep({ref}) > refStep({rfk})"
            msgs[key] = msgs.get(key, 0) + 1
    return [f"{msg} x{cnt}" if cnt > 1 else msg for msg, cnt in msgs.items()] or ["(see constraint list in system prompt)"]


# ─────────────────────────────────────────────────────────────────────────────
# Frontier gap detection
# ─────────────────────────────────────────────────────────────────────────────

def _frontier_gaps(results: list[RoundResult], best_cov: float, best_time: float) -> list[str]:
    """Identify promising unexplored regions of the coverage/time front."""
    gaps = []
    # Gap 1: high coverage but slow — no sample with time < median AND cov > threshold
    if best_cov >= SC.TCS_HIGH_COV_THRESHOLD:
        times = sorted(r.mean_time for r in results)
        median_time = times[len(times) // 2] if times else 9999.0
        fast_high = any(r.mean_time < median_time and r.coverage > SC.TCS_HIGH_COV_THRESHOLD for r in results)
        if not fast_high:
            gaps.append(f"No sample with time<{median_time:.2f}s AND cov>{SC.TCS_HIGH_COV_THRESHOLD:.2f} -- explore faster configs")
    # Gap 2: best coverage still below target
    if best_cov < SC.TARGET_COVERAGE:
        gaps.append(f"Best coverage {best_cov:.2f} below {SC.TARGET_COVERAGE:.2f} -- prioritise higher coverage")
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_KEY_PARAMS = ["refStep", "distQuantification", "angleQuantification",
               "referredStep", "maxVoteRatio", "outputNum",
               "operationApproach", "deviationCorrectionCapacity"]

def _short_params(params: dict) -> str:
    parts = []
    for k in _KEY_PARAMS:
        if k in params:
            v = params[k]
            short_k = {
                "refStep": "refSt",
                "distQuantification": "distQ",
                "angleQuantification": "angQ",
                "referredStep": "refRd",
                "maxVoteRatio": "vote",
                "outputNum": "outN",
                "operationApproach": "opApp",
                "deviationCorrectionCapacity": "devCap",
            }.get(k, k[:6])
            if isinstance(v, float):
                parts.append(f"{short_k}={v:.2f}")
            else:
                parts.append(f"{short_k}={v}")
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = [
        RoundResult(0, {"refStep": 10, "distQuantification": 1.0, "angleQuantification": 60,
                        "referredStep": 5, "maxVoteRatio": 0.7, "outputNum": 1,
                        "operationApproach": 1, "deviationCorrectionCapacity": 1},
                    coverage=0.72, mean_time=0.55),
        RoundResult(1, {"refStep": 6,  "distQuantification": 2.0, "angleQuantification": 60,
                        "referredStep": 5, "maxVoteRatio": 0.6, "outputNum": 1,
                        "operationApproach": 1, "deviationCorrectionCapacity": 1},
                    coverage=0.84, mean_time=0.31),
        RoundResult(2, {"refStep": 8,  "distQuantification": 1.5, "angleQuantification": 90,
                        "referredStep": 9, "maxVoteRatio": 0.7, "outputNum": 2,
                        "operationApproach": 2, "deviationCorrectionCapacity": 1},
                    coverage=0.0, mean_time=0.0, violated=True),
        RoundResult(3, {"refStep": 12, "distQuantification": 0.75, "angleQuantification": 60,
                        "referredStep": 6, "maxVoteRatio": 0.8, "outputNum": 1,
                        "operationApproach": 2, "deviationCorrectionCapacity": 0},
                    coverage=0.91, mean_time=0.88),
    ]
    print(summarize(results))
