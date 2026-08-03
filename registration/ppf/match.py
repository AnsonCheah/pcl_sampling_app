"""Voting, peak extraction, pose clustering and verification.

One segmented instance cluster in, a ranked list of poses out.  Matching a cluster rather
than the whole bin is what makes this affordable: measured on a 26-part bunny scene, the
whole-scene formulation costs ~50 M scene pair-evaluations extrapolated to 100 parts, while
per-instance costs ~4 M, because a cluster's points only ever pair with each other.

The accumulator is Drost's: one 2-D table per scene reference point, indexed by (model
point, alpha bin).  A vote for cell ``(m, a)`` says "if this scene reference point is model
point ``m``, rotated by ``a``, then this pair is explained".  The tallest cell wins.

Two departures from textbook Drost, both to stop non-discriminative geometry dominating:

* **Vote deduplication** (Hinterstoisser, ECCV 2016).  On a planar patch one scene point can
  match hundreds of equivalent model points, all voting the same cell, so a flat region
  out-votes a distinctive one purely by repetition.  Each scene point is therefore counted
  at most once per cell.
* **Per-model-point vote weights.**  The vote is a weight rather than a ``+1``, which is the
  mechanism the ambiguity heat map and the PPF-saliency map plug into.  Uniform weights
  reproduce the unweighted result exactly, and there is a test pinning that.

Note that deduplication does **not** subsume MechVision's ``maxNumOfPointPairsPerFeature``,
which was an early assumption here and is wrong: dedup runs *after* the table lookup has
been expanded, so it corrects the bias but not the cost.  Bounding the cost needs a cap at
the source, which lives in ``PPFConfig.max_bucket_entries`` and is applied when the model is
trained.  Batching here is a second, independent guard on peak memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ._frames import alpha_of, frames_to_x, pose_from_correspondence
from .config import PPFConfig
from .model import PPFModel, pair_features

__all__ = ["Pose", "MatchResult", "match", "downsample"]

_TWO_PI = 2.0 * np.pi


@dataclass
class Pose:
    """One returned hypothesis, model frame -> scene frame."""

    T: np.ndarray                    # (4, 4)
    votes: float                     # summed accumulator weight over the cluster
    score: float                     # fraction of scene points explained; the accept gate
    n_support: int                   # distinct model points landing on scene points
    peak_votes: float = 0.0          # tallest single accumulator cell in this cluster
    runner_up_votes: float = 0.0     # tallest cell outside the peak's alpha neighbourhood
    n_members: int = 1               # hypotheses merged into this cluster

    @property
    def peak_margin(self) -> float:
        """``1 - runner_up/peak``, in [0, 1]. High = the accumulator had a clear winner.

        Reported because the literature never has: no PPF paper relates accumulator health
        to model sparsity, so the sparsity arms of the ablation have nothing to compare
        against.  A pose that wins by a hair on a sparsified model is a different outcome
        from one that wins outright, even when both land within tolerance.
        """
        if self.peak_votes <= 0:
            return 0.0
        return float(1.0 - self.runner_up_votes / self.peak_votes)


@dataclass
class MatchResult:
    poses: List[Pose] = field(default_factory=list)
    n_scene_points: int = 0          # after downsampling
    n_pair_evals: int = 0            # scene pairs actually looked up
    n_votes: int = 0                 # votes cast after dedup
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> Optional[Pose]:
        return self.poses[0] if self.poses else None


# ----------------------------------------------------------------------
def downsample(points: np.ndarray, normals: np.ndarray, voxel: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel-average a cloud, keeping normals unit.

    The scene is downsampled at the *model's* tau, not at some independently chosen
    resolution: PPF's distance bins are shared between model and scene, so mismatched
    densities put the same physical pair in different bins on each side.

    Open3D does the binning (12x faster than doing it here with ``np.unique(axis=0)`` plus
    ``np.add.at``, both of which are slow paths in NumPy). What Open3D does *not* do is
    renormalise the averaged normals — they come back up to ~0.2% off unit — and PPF's
    features are angles between normals, so that is left to us.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if voxel <= 0 or len(pts) == 0:
        return pts, nrm

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    down = pcd.voxel_down_sample(voxel)

    out_p = np.asarray(down.points, dtype=np.float64)
    out_n = np.asarray(down.normals, dtype=np.float64)
    mag = np.linalg.norm(out_n, axis=1)
    # A voxel straddling a thin wall averages two opposing normals to ~zero; that point has
    # no usable orientation, so drop it rather than emit an arbitrary direction.
    keep = mag > 1e-9
    return out_p[keep], out_n[keep] / mag[keep, None]


def _pair_weights(model: PPFModel, entry: np.ndarray, mode: str) -> Optional[np.ndarray]:
    """Vote weight per matched model-table entry, or ``None`` for unweighted."""
    w = model.weights
    if w is None:
        return None
    if mode == "ref":
        return w[model.entry_point[entry]]
    w1 = w[model.entry_point[entry]]
    w2 = w[model.entry_point2[entry]]
    if mode == "product":
        return w1 * w2
    if mode == "geometric_mean":
        return np.sqrt(w1 * w2)
    raise ValueError(f"unknown weight mode {mode!r}")


def _rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotations, broadcasting over leading axes."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)          # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _cluster(T: np.ndarray, votes: np.ndarray, peaks: np.ndarray, runners: np.ndarray,
             pos_tol: float, ang_tol_deg: float, max_clusters: int = 64):
    """Greedy SE(3) agglomeration, strongest hypothesis first.

    Both position *and* rotation are compared.  Translation-only clustering — which is what
    the previous ``coarse_match._distance_nms`` did — merges two genuinely different
    orientations of the same part at the same location, which is exactly the symmetry-flip
    case this project is trying to measure.  Collapsing it would hide the failure.
    """
    from geometry.geom_utils import project_to_so3

    order = np.argsort(-votes)
    reps: List[np.ndarray] = []
    members: List[List[int]] = []
    for i in order:
        Ti = T[i]
        for c, rep in enumerate(reps):
            if (np.linalg.norm(Ti[:3, 3] - rep[:3, 3]) < pos_tol
                    and _rotation_angle_deg(rep[:3, :3], Ti[:3, :3]) < ang_tol_deg):
                members[c].append(i)
                break
        else:
            if len(reps) >= max_clusters:
                continue
            reps.append(Ti)
            members.append([i])

    out = []
    for c, mem in enumerate(members):
        idx = np.asarray(mem)
        Tm = np.eye(4)
        # Averaging rotation matrices leaves SO(3); without the projection the transform
        # quietly scales and shears the model and verification scores it as a near-miss.
        Tm[:3, :3] = project_to_so3(T[idx, :3, :3].mean(axis=0))
        Tm[:3, 3] = T[idx, :3, 3].mean(axis=0)
        out.append((Tm, float(votes[idx].sum()), float(peaks[idx].max()),
                    float(runners[idx].max()), len(idx)))
    return out


def _verify(model: PPFModel, T: np.ndarray, scene_pts: np.ndarray, scene_nrm: np.ndarray,
            scene_tree: cKDTree, tol: float, cos_tol: float) -> Tuple[float, int]:
    """Score a pose by how much of the *scene* cluster it explains.

    Scored scene-side rather than model-side on purpose: instances are 10-48% visible in a
    real bin, so a correct pose necessarily leaves most of the model unmatched and a
    model-side score would penalise it for being occluded.  Every point in the segmented
    cluster, on the other hand, ought to lie on the part.

    **Normal agreement is required, not just proximity.** Distance alone is close to useless
    as a tie-breaker on a rounded or symmetric part: a pose with the symmetry axis tilted the
    wrong way still drapes model surface near every scene point and scores as well as the
    correct one. Measured on T-LESS, adding this and halving the tolerance moved top-1
    selection from 0.75 to 1.00 on obj_000017 and 0.28 to 0.40 on obj_000013, and never hurt.

    That matters more than it sounds, because the failure it fixes is a *ranking* failure,
    not a search failure: taking the best of the top 10 hypotheses instead of the first
    reaches 0.88 on obj_000001 where the top-1 reaches 0.28. The right pose is usually
    already in the candidate set; the score just has to pick it out.
    """
    moved = model.points @ T[:3, :3].T + T[:3, 3]
    moved_n = model.normals @ T[:3, :3].T
    d_scene, i_scene = cKDTree(moved).query(scene_pts, k=1)
    ok = (d_scene < tol) & (np.einsum("ij,ij->i", scene_nrm, moved_n[i_scene]) > cos_tol)
    d_model, _ = scene_tree.query(moved, k=1)
    return float(np.mean(ok)), int(np.count_nonzero(d_model < tol))


# ----------------------------------------------------------------------
def match(model: PPFModel,
          points: np.ndarray,
          normals: np.ndarray,
          top_k: int = 1,
          weight_mode: str = "ref",
          do_downsample: bool = True,
          vote_budget: int = 8_000_000,
          accumulator_cells: int = 4_000_000,
          backend: str = "numpy") -> MatchResult:
    """Match ``model`` against one segmented instance cluster.

    Parameters
    ----------
    points, normals : the cluster, in scene coordinates. Raw resolution is fine — see
        ``do_downsample``.
    top_k : hypotheses to return, best first. The ambiguity study wants >1 so that a
        symmetry flip and the true pose can both be seen.
    weight_mode : how a pair's two endpoint weights combine (``ref`` / ``product`` /
        ``geometric_mean``). Ignored when the model carries no weights.
    vote_budget, accumulator_cells : memory budgets in elements. Hardware policy shared by
        every part, not per-part tuning — they bound peak RAM on degenerate (planar) parts
        where one feature bin holds a large share of the model's pairs.
    """
    import time

    if backend != "numpy":
        # Kept in the signature so callers and tests are written against the final API.
        raise NotImplementedError(
            f"backend {backend!r} is not implemented yet; the CuPy kernel is Phase 4 and "
            f"the NumPy path is its correctness oracle")

    cfg: PPFConfig = model.cfg
    t0 = time.perf_counter()
    result = MatchResult()

    s_pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    s_nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    s_nrm = s_nrm / np.maximum(np.linalg.norm(s_nrm, axis=1, keepdims=True), 1e-12)
    if do_downsample:
        s_pts, s_nrm = downsample(s_pts, s_nrm, cfg.tau)
    C = len(s_pts)
    result.n_scene_points = C
    if C < 3:
        return result

    s_frames = frames_to_x(s_nrm)
    M, NA = model.n_points, cfg.n_alpha
    cells = M * NA
    astep = _TWO_PI / NA
    ref_chunk = max(1, min(C, int(accumulator_cells // max(cells, 1))))

    all_T, all_votes, all_peak, all_runner = [], [], [], []
    t_vote = 0.0

    for c0 in range(0, C, ref_chunk):
        c1 = min(c0 + ref_chunk, C)
        R = c1 - c0
        acc = np.zeros(R * cells, dtype=np.float64)

        local = np.repeat(np.arange(R), C)
        ii = np.repeat(np.arange(c0, c1), C)
        jj = np.tile(np.arange(C), R)
        keep = ii != jj
        local, ii, jj = local[keep], ii[keep], jj[keep]

        dist, f2, f3, f4 = pair_features(s_pts, s_nrm, ii, jj)
        usable = (dist >= cfg.min_pair_dist) & (dist <= cfg.max_pair_dist)
        if cfg.steep_pair_readmit_deg > 0:
            cos_lim = np.cos(np.deg2rad(cfg.steep_pair_readmit_deg))
            usable |= ((np.einsum("ij,ij->i", s_nrm[ii], s_nrm[jj]) < cos_lim)
                       & (dist <= cfg.max_pair_dist) & (dist > 1e-12))
        local, ii, jj = local[usable], ii[usable], jj[usable]
        if len(ii) == 0:
            continue
        keys = model.quantise(dist[usable], f2[usable], f3[usable], f4[usable])
        alpha_s = alpha_of(s_frames[ii], s_pts[ii], s_pts[jj])

        lo, hi = model.lookup(keys)
        counts = (hi - lo).astype(np.int64)
        hit = counts > 0
        if not hit.any():
            continue
        local, jj, alpha_s = local[hit], jj[hit], alpha_s[hit]
        lo, counts = lo[hit], counts[hit]
        result.n_pair_evals += int(hit.sum())

        # Split into batches whose expanded vote count stays inside the budget. A single
        # over-budget bucket still goes through as its own batch rather than failing.
        edges = _budget_batches(counts, vote_budget)
        tv = time.perf_counter()
        for b0, b1 in edges:
            cnt = counts[b0:b1]
            total = int(cnt.sum())
            if total == 0:
                continue
            starts = np.cumsum(cnt) - cnt
            src = np.repeat(np.arange(b0, b1), cnt)
            entry = (np.repeat(lo[b0:b1] - starts, cnt) + np.arange(total))

            alpha = model.entry_alpha[entry].astype(np.float64) - alpha_s[src]
            abin = np.floor((alpha % _TWO_PI) / astep).astype(np.int64) % NA
            cell = (local[src] * M + model.entry_point[entry]) * NA + abin
            w = _pair_weights(model, entry, weight_mode)

            if cfg.vote_dedup:
                # One vote per (cell, scene point). Without this a scene point on a planar
                # patch votes once per equivalent model point, and flat geometry outvotes
                # distinctive geometry by sheer repetition.
                cell, w = _dedup(cell * C + jj[src], C, w)
            acc += np.bincount(cell, weights=w, minlength=R * cells)
            result.n_votes += len(cell)
        t_vote += time.perf_counter() - tv

        flat = acc.reshape(R, cells)
        best = flat.argmax(axis=1)
        rows = np.arange(R)
        peak = flat[rows, best]
        alive = peak > 0
        if not alive.any():
            continue

        m_idx, a_idx = best // NA, best % NA
        # Runner-up excludes the winner's own alpha neighbourhood, which is where vote
        # spreading deposits copies of the winner -- counting those would report a tiny
        # margin for every clean detection.
        tmp = flat.copy()
        for off in (-1, 0, 1):
            tmp[rows, m_idx * NA + (a_idx + off) % NA] = -1.0
        runner = np.maximum(tmp.max(axis=1), 0.0)

        r = rows[alive]
        refs = np.arange(c0, c1)[alive]
        T = pose_from_correspondence(
            model.points[m_idx[alive]], model.frames[m_idx[alive]],
            s_pts[refs], s_frames[refs], (a_idx[alive] + 0.5) * astep)
        all_T.append(T)
        all_votes.append(peak[r])
        all_peak.append(peak[r])
        all_runner.append(runner[r])

    result.timings["vote"] = t_vote
    if not all_T:
        result.timings["total"] = time.perf_counter() - t0
        return result

    T = np.concatenate(all_T)
    votes = np.concatenate(all_votes)
    peaks = np.concatenate(all_peak)
    runners = np.concatenate(all_runner)

    clusters = _cluster(T, votes, peaks, runners,
                        cfg.cluster_pos_tol, cfg.cluster_ang_tol_deg)
    tree = cKDTree(s_pts)
    poses: List[Pose] = []
    for Tc, v, pk, ru, n_mem in clusters[: max(top_k * 8, 16)]:
        score, n_sup = _verify(model, Tc, s_pts, s_nrm, tree,
                               cfg.verify_tol, cfg.verify_cos_tol)
        if score < cfg.accept_score:
            continue
        poses.append(Pose(T=Tc, votes=v, score=score, n_support=n_sup,
                          peak_votes=pk, runner_up_votes=ru, n_members=n_mem))

    # Rank by verified coverage, not by raw votes: votes measure how loudly the accumulator
    # agreed, coverage measures whether the pose is actually consistent with the observation,
    # and on ambiguous parts the loudest hypothesis is routinely the flipped one.
    poses.sort(key=lambda p: (-p.score, -p.votes))
    result.poses = poses[:top_k]
    result.timings["total"] = time.perf_counter() - t0
    return result


def _dedup(comp: np.ndarray, divisor: int, w: Optional[np.ndarray]):
    """Collapse duplicate ``comp`` values, returning ``(comp // divisor, weights)``.

    Deliberately sort-based rather than ``np.unique``.  NumPy 2.x routes ``unique`` through
    a hash table which is pathologically slow on the wide int64 keys used here: measured
    0.411 s versus 0.020 s for sort-then-diff on 1.4 M values, and this is the single
    hottest call in the matcher (53% of runtime before the change).
    """
    if w is None:
        s = np.sort(comp)
        keep = np.empty(len(s), dtype=bool)
        keep[0] = True
        np.not_equal(s[1:], s[:-1], out=keep[1:])
        return s[keep] // divisor, None
    order = np.argsort(comp, kind="stable")
    s = comp[order]
    keep = np.empty(len(s), dtype=bool)
    keep[0] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep] // divisor, w[order][keep]


def _budget_batches(counts: np.ndarray, budget: int) -> List[Tuple[int, int]]:
    """Contiguous index ranges over ``counts`` whose sums stay under ``budget``.

    Cut points come from ``searchsorted`` on the cumulative sum rather than a Python loop
    over ``counts``. This only fires on degenerate parts — where the total blows the budget
    — but those are exactly the parts with the most pairs, so the loop was slowest precisely
    when it ran: 43 ms over 250 k pairs.

    A single bucket larger than the budget still becomes its own batch rather than failing;
    the budget is a guard against exhausting memory, not a hard contract.
    """
    total = int(counts.sum())
    if total <= budget:
        return [(0, len(counts))]

    cum = np.cumsum(counts)
    out: List[Tuple[int, int]] = []
    start = 0
    while start < len(counts):
        base = cum[start - 1] if start else 0
        # Last index whose running total still fits; +1 guarantees forward progress when a
        # single entry already exceeds the budget.
        end = int(np.searchsorted(cum, base + budget, side="right"))
        end = max(end, start + 1)
        out.append((start, end))
        start = end
    return out
