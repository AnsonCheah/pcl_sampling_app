"""Per-model-point vote weights.

Two independent weight sources, because they fail in different places.

**Ambiguity saliency** (``transfer_weights`` applied to
``AmbiguityProfile.per_point_discriminative``) asks "could this point be somewhere else on
the model?".  It is the sharper signal when it exists, but it is derived purely from the
recovered ambiguity axes, so on a part where none are found it is uniformly 1.0 and carries
no information at all.  Measured on the Stanford bunny: zero axes, every point scored 1.000.
It also saturates the other way -- ``test_part.stl`` scores ``discriminative_fraction=0.015``,
meaning 98.5% of its surface is explained by *some* ambiguity transform, so a hard threshold
there would keep ~2% of the cloud.

**PPF saliency** (``ppf_saliency``) asks a narrower question -- "do this point's pairs look
like everybody else's?" -- and is always defined, needs no symmetry to have been found, and
costs nothing because it reads the table the matcher already built.  It is the
descriptor-specific analogue of keypoint saliency: Salti et al. (ICCV 2015) argued detectors
should score points by whether *a given descriptor* can match them rather than by generic
geometric distinctiveness, and this instantiates that for PPF, which nobody appears to have
done.

Both are returned in [0, 1] and are meant to be handed to ``PPFModel.with_weights``.  They
compose by multiplication; the ablation measures whether that helps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["ppf_saliency", "transfer_weights", "combine"]


def ppf_saliency(model, normalise: str = "rank", eps: float = 1e-12) -> np.ndarray:
    """Per-model-point weight from how populated its feature bins are.

    A pair whose quantised feature lands in a bin holding ``n`` of the model's ``E`` entries
    carries ``-log(n/E)`` nats about which pose is correct: a bin holding one pair pins the
    pose down, a bin holding a thousand says almost nothing.  Averaging that over the pairs a
    point participates in scores the point.

    This is exactly the quantity that collapses on the geometry PPF is known to fail on.  A
    planar patch puts a huge share of its pairs in a handful of bins, so every point on it
    scores low -- which is the same failure Hinterstoisser's vote deduplication attacks from
    the other end, and Vidal's normal-aware clustering attacks from a third.

    Parameters
    ----------
    normalise : ``"rank"`` maps scores to their within-model quantiles, giving a weight
        distribution that does not depend on the part's absolute feature entropy -- the point
        being that a weight of 0.5 should mean "median for this part" for every part, since
        anything else reintroduces per-part calibration.  ``"minmax"`` keeps the raw shape.
    """
    keys = model.keys
    E = len(keys)
    if E == 0:
        return np.ones(model.n_points)

    # Run lengths over the sorted key array give each entry its bin population.
    first = np.searchsorted(keys, keys, side="left")
    count = np.searchsorted(keys, keys, side="right") - first
    info = -np.log(np.maximum(count / E, eps))

    owner = model.entry_point
    total = np.bincount(owner, weights=info, minlength=model.n_points)
    n = np.bincount(owner, minlength=model.n_points)
    # A point with no surviving pairs cannot disambiguate anything; score it at the floor
    # rather than dropping it, so the weight array stays index-aligned with the model.
    score = np.where(n > 0, total / np.maximum(n, 1), 0.0)
    return _normalise(score, normalise)


def transfer_weights(src_points: np.ndarray, src_values: np.ndarray,
                     dst_points: np.ndarray, normalise: Optional[str] = None) -> np.ndarray:
    """Carry a per-point scalar from one cloud onto another by nearest neighbour.

    Needed because the ambiguity heat map is computed on the full reference cloud (~20 k
    points) while the PPF model is voxel-downsampled from it (~500), so the two are not
    index-aligned however tempting the assumption is.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 3)
    vals = np.asarray(src_values, dtype=np.float64).ravel()
    if len(vals) != len(src):
        raise ValueError(f"{len(src)} source points but {len(vals)} values")
    out = vals[cKDTree(src).query(dst, k=1)[1]]
    return _normalise(out, normalise) if normalise else out


def combine(*weights: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Multiply weight sources, holding the result off zero.

    The floor matters more than it looks.  A weight of exactly zero does not down-rank a
    model point, it deletes it -- and deleting points is the pruning strategy the evidence
    argues against, since PPF builds all O(N^2) pairs and losing a point costs its whole row.
    Keeping a small floor means a weighted arm stays a *weighting* arm rather than quietly
    turning into a pruning arm at the low end.
    """
    if not weights:
        raise ValueError("combine() needs at least one weight array")
    out = np.ones_like(np.asarray(weights[0], dtype=np.float64))
    for w in weights:
        w = np.asarray(w, dtype=np.float64).ravel()
        if len(w) != len(out):
            raise ValueError("weight arrays must be the same length")
        out = out * w
    return np.clip(out, floor, None)


def _normalise(x: np.ndarray, how: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if how == "rank":
        if len(x) < 2:
            return np.ones_like(x)
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
        return ranks / (len(x) - 1)
    if how == "minmax":
        lo, hi = float(x.min()), float(x.max())
        return np.ones_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)
    raise ValueError(f"unknown normalisation {how!r}")
