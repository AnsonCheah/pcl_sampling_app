# Plan: Transposition + Look-ahead for MM_Optimizer

## Context

Pascal Pons' Connect4 solver (blog.gamesolver.org) demonstrates that **alpha-beta pruning** (look-ahead) and **transposition tables** (memoization) are both individually exponential in cost, but when combined they attack the exponential from opposite ends — pruning cuts width (branches to explore), caching cuts depth redundancy (paths that re-visit the same state). Together they make the effective search tree orders of magnitude smaller.

The user's insight: MM_Optimizer has analogous structure. The full Cartesian product of ~25 parameters is ~40,000+ combinations per part. The existing 6-phase hierarchical coordinate descent plan already constrains the search, but it currently:
1. Re-evaluates the same config in multiple phases (Phase 5 re-sweeps what Phase 2b already tested)
2. Runs full M=30 scene evaluations on clearly bad candidates
3. Enters later phases without bounding whether they can possibly improve on current best

The four strategies below are **orthogonal additions** to `plan_1_hierarchical_decent.md` — the phase structure remains the algorithmic backbone; these reduce the evaluation budget without changing the search logic.

---

## Conceptual Mapping

| Connect4 concept | MM_Optimizer analog |
|---|---|
| Board position (hash) | `(config_hash, scene_set_hash)` |
| Transposition — same position via different paths | Same config re-tested in Phase 5 that was already evaluated in Phase 2b |
| Alpha — best lower bound found so far | `best_coverage_so_far` across all evaluations to date |
| Pruned branch — can't beat alpha | Configs that fail M_small=5 cheaply, before spending M_full=30 |
| Opening book — pre-solved positions | Cached `(geometry_vector → best_params)` from prior parts |
| Branching factor | # candidates per parameter sweep |
| Tree depth | Phase chain: 2 → 3 → 4 → 5 → 6 |

---

## Strategy 1: Exact Config Cache (Transposition Table)

**Problem**: Phase 5 re-sweeps `maxVoteRatio`, `outputNum`, `referredStep` — all evaluated in Phase 2b. Phase 6 interval narrowing may land on Phase 2a grid points. Estimated ~30–50 duplicate evaluations across a full run. Currently no deduplication exists.

**Solution**: `EvalCache` — a dict keyed by `(config_hash, scene_set_hash)`. Every `evaluate_config()` call checks the cache first. Serialized to disk for crash resume and cross-run reuse.

```python
# Key construction
key = sha256(json.dumps(sorted(config.items())) + "::" + ",".join(sorted(scene_paths)))[:16]
cache[key] = {"score": ..., "coverage": ..., "mean_time": ..., "per_run": [...]}
```

**Where**: Wrap `evaluate_config()` in `optimizer.py` — the rest of the phases call through unchanged.

**Benefit**: ~15–20% fewer evaluations at zero quality cost. Free crash-resume.

---

## Strategy 2: Two-Pass Multi-Fidelity (Alpha-Beta Analog)

**The core "both curves" insight**: In alpha-beta, you establish a bound cheaply before committing to full subtree expansion. Here, you evaluate all N candidates cheaply (M_small=5 scenes) to rank them, then run full M_full=30 only on the top-K survivors.

- **Width control (pruning)**: Pass 1 on M_small=5 eliminates most bad candidates cheaply
- **Depth control (convergence)**: Pass 2 on M_full=30 only for the K=3 survivors

**Protocol**:
```
evaluate_phase_sweep(candidates, scenes, M_small=5, M_full=30, K=3):
    pass1 = [evaluate(c, scenes[:M_small]) for c in candidates]  # cheap
    top_k = sorted(pass1, by=score)[:K]
    pass2 = [evaluate(c, scenes) for c in top_k]                 # full, cache-stored
    return best(pass2)
```

Pass 1 results are also cached (key = config_hash + subset_hash). If a Pass 1 survivor was already evaluated at M_full in a prior phase, cache hit avoids Pass 2 entirely.

**Phase-specific numbers (M_full=30)**:

| Phase | N candidates | Pass 1 evals | K→Pass 2 evals | Old cost | New cost | Savings |
|---|---|---|---|---|---|---|
| 2a joint grid | 20 | 20×5=100 | 3×30=90 | 20×30=600 | 190 | **68%** |
| 2b per-param (avg 5 candidates) | 5 | 5×5=25 | 2×30=60 | 150 | 85 | **43%** |
| 3 per-param (avg 4 candidates) | 4 | 4×5=20 | 2×30=60 | 120 | 80 | **33%** |
| 5 joint refinement | 12 | 12×5=60 | 3×30=90 | 360 | 150 | **58%** |

**Special case — boolean sweeps** (2 candidates): skip two-pass, evaluate both at M_full directly. Overhead not worth it.

**Total budget**: ~200 evaluations (current plan) → **~80–100 evaluations** with strategies 1+2 combined.

**Correctness**: K=3 survivors from Pass 2 feed into Phase 6 interval narrowing (already in the plan). The dense ±search in Phase 6 recovers any fine-grained optimum that Pass 1 noise might have suppressed.

---

## Strategy 3: Phase-Level Bounds Pruning (Macro Alpha-Beta)

After each phase completes, check a bound before entering the next phase. This is alpha-beta at the granularity of entire phases rather than individual evaluations.

```
After Phase 1:  best_loose_coverage < 0.50  →  STOP, report "not tunable for this part"
After Phase 2a: best_tight_coverage < 0.30  →  WARN: PPF quantization may be structurally wrong;
                                                skip Phase 2b, log geometry analysis for review
After Phase 2:  best_tight_coverage < 0.65  →  skip Phase 4 (symmetry correction on a poorly-
                                                tuned baseline has negligible expected benefit)
After Phase 3:  best_tight_coverage < 0.75  →  skip Phase 6 (interval narrowing not justified);
                                                write best-found config and exit
```

**Coarse-to-fine look-ahead** (within Phase 3): Use Phase 2's best coarse pose errors to pre-select `operationApproach` candidates before Phase 3 begins — no extra evaluation needed, data already computed.

```python
median_coarse_err = median(pos_errors from Phase 2 best config's per-run results)

if median_coarse_err < 0.003:   candidates = [HighSpeed, Standard]
elif median_coarse_err < 0.010: candidates = [Standard, HighAccuracy]
else:                           candidates = [HighAccuracy, ExtraHighAccuracy]
```

This halves the first sweep in Phase 3 (Priority 1: `operationApproach`) without quality loss. The logic matches `plan_1_hierarchical_decent.md`'s coupling rule: "well-tuned coarse → can use HighSpeed; loose coarse → needs HighAccuracy."

---

## Strategy 4: Cross-Part Opening Book (Long-Horizon, Deferred)

**Concept**: After each successful optimization, append `{geometry_vector, best_params, part_name}` to `opening_book.json`. When optimizing a new part, compute geometry features and query k-nearest neighbors. If the nearest neighbor's geometry distance is below a confidence threshold, **start Phase 2 from the matched params** instead of the geometric warm-start. Potentially skip Phase 2a entirely.

**Geometry feature vector** (normalized):
```
[log10(diameter), flatness_ratio, normal_concentration, SA/D², sym_order_hint]
```

**Lookup**:
```python
hits = knn_query(new_features, opening_book, k=3)
if hits[0].distance < MATCH_THRESHOLD:
    warm_params = hits[0].best_params  # empirical warm-start
    skip_phase_2a = True               # already near-optimal quantization
```

**Why this is the most powerful long-term strategy**: As the book grows past ~50 parts, the average evaluation budget per part drops from ~100 to potentially ~20–30 (Phase 5 + 6 only). This is exactly the Connect4 solver's opening book effect — the book covers the opening, the search handles only the unique "endgame" of each new part.

**Implementation**: `MM_Optimizer/opening_book.py` — deferred until the main optimizer exists and has produced results on 10+ parts.

---

## Integration with plan_1_hierarchical_decent.md

`plan_1_hierarchical_decent.md` is unchanged in structure. These strategies are implemented as:
- `EvalCache` class (Strategy 1) — wraps `evaluate_config()`
- `evaluate_phase_sweep()` (Strategy 2) — called by each phase instead of bare `evaluate_config()` loops
- `phase_gate()` checks (Strategy 3) — inserted at phase transitions in the main loop
- `opening_book.py` (Strategy 4) — separate module, called at Phase 0 and post-optimization

---

## Files to Create / Modify

| Action | File | What |
|---|---|---|
| **Create** | `MM_Optimizer/optimizer.py` | Add `EvalCache`, `evaluate_phase_sweep()`, `phase_gate()` as first-class constructs alongside the 6-phase loop |
| **Create** | `MM_Optimizer/opening_book.py` | Deferred — after 10+ parts have been optimized |
| **Reference** | `MM_Optimizer/plan_1_hierarchical_decent.md` | Unchanged — this plan supplements it |

---

## Verification

1. **Cache correctness**: Run Phase 2b sweep twice on same candidates. Second run should return 100% cache hits, zero MechVision calls.
2. **Two-pass fidelity**: On a known part, compare Pass 1 ranking (M_small=5) vs Pass 2 ranking (M_full=30). K=3 survivors should contain the true optimum in >90% of sweeps.
3. **Phase gate**: Run on a degenerate scene set (all black, no geometry). Expect Phase 1 gate to fire at < 0.50 coverage, clean exit with "not tunable" message.
4. **Coarse look-ahead**: Log `median_coarse_err` and resulting `operationApproach` candidates before Phase 3. Verify the chosen candidate range contains the Phase 3 winner.
5. **Full budget audit**: After a complete run, print a breakdown of: total evals called, cache hits, Phase 1 prunes, Phase 2/3 prunes. Target: ≤ 110 MechVision calls for a "normal" part.
