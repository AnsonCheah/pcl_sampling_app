"""The trained PPF model: quantised point-pair features in a sorted-key lookup table.

Training enumerates every ordered pair of model points, reduces each to the four-number
Drost feature, quantises it to an integer key, and stores ``(model_point_index, alpha)``
under that key.  Matching then reduces a scene pair the same way and looks the key up.

Two implementation choices worth stating, because both are easy to get wrong:

* **A sorted array with binary search, not a Python dict.**  The table has ~250 k entries
  for a 500-point model and is queried ~160 k times per instance; a dict costs a Python
  object per probe and cannot be handed to a GPU at all.  Sorted keys plus ``searchsorted``
  is vectorised, cache-friendly, and ports to CuPy unchanged.
* **Feature-bin spreading happens here, at train time, not at match time.**  Sensor noise
  can push a correct correspondence into an adjacent bin, so each pair is also stored under
  its neighbours.  Doing it on the model costs memory once and is then reused across all
  ~100 instances in a bin; doing it on the scene would multiply every lookup instead.

The alpha angle is stored as a float, not pre-binned.  Binning it here would throw away
precision before the only place it is needed -- the difference ``alpha_model - alpha_scene``
-- and that difference is what sets the recovered rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .._shared._frames import alpha_of, frames_to_x
from .config import PPFConfig

__all__ = ["PPFModel"]


# Spreading offsets applied to (f_dist, f4). The distance bin and the normal-to-normal
# angle are the two most noise-sensitive components -- f2/f3 involve the pair direction,
# which is far better conditioned than either. Spreading all four (Hinterstoisser's 80
# neighbours) is available via `spread="full"` but costs 9x the memory of this default.
_SPREAD_9 = np.array([(a, 0, 0, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], dtype=np.int64)
_SPREAD_81 = np.array([(a, b, c, d)
                       for a in (-1, 0, 1) for b in (-1, 0, 1)
                       for c in (-1, 0, 1) for d in (-1, 0, 1)], dtype=np.int64)


@dataclass
class PPFModel:
    """A trained model. Build with :meth:`train`."""

    points: np.ndarray               # (M, 3) sampled model points
    normals: np.ndarray              # (M, 3) unit
    frames: np.ndarray               # (M, 3, 3) local frames, R @ n = +x
    cfg: PPFConfig

    keys: np.ndarray                 # (E,) int64, sorted
    entry_point: np.ndarray          # (E,) int32 -- index into `points` of the pair's FIRST point
    # The pair's SECOND point. Not needed to build a pose (the accumulator axis is the first
    # point), but the `product` / `geometric_mean` weight modes weight a pair by both of its
    # endpoints, so the ablation cannot run without it. 4 bytes/entry.
    entry_point2: np.ndarray         # (E,) int32
    entry_alpha: np.ndarray          # (E,) float32
    n_dist_bins: int
    n_pairs: int                     # pairs before spreading

    weights: Optional[np.ndarray] = None   # (M,) per-model-point vote weight, or None

    # CSR-style direct index over the whole key space: `key_offsets[k]` is where bin `k`
    # starts in the entry arrays. None when the key space was too large to materialise, in
    # which case `lookup` falls back to binary search. See `_build_key_offsets`.
    # Carried through `with_weights` automatically -- it rebuilds from `self.__dict__`.
    key_offsets: Optional[np.ndarray] = None

    # cKDTree over `points` in the MODEL frame, built once at train time and queried by
    # _verify. Read-only, so it is safe to share across threads.
    point_tree: Optional[object] = None

    # Device copies of the table arrays, keyed by array-module name; see `table_for`.
    device_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return len(self.points)

    def table_for(self, xp):
        """``(keys, entry_point, entry_point2, entry_alpha, key_offsets)`` on ``xp``'s device.

        Returns the host arrays untouched for NumPy, so the CPU path costs nothing. The table
        is ~40 MB and constant, so it is transferred once per model and reused.
        """
        if xp is np:
            return (self.keys, self.entry_point, self.entry_point2, self.entry_alpha,
                    self.key_offsets)
        if self.device_cache is None:
            self.device_cache = {}
        name = xp.__name__
        if name not in self.device_cache:
            self.device_cache[name] = (
                xp.asarray(self.keys), xp.asarray(self.entry_point),
                xp.asarray(self.entry_point2), xp.asarray(self.entry_alpha),
                None if self.key_offsets is None else xp.asarray(self.key_offsets))
        return self.device_cache[name]

    def with_weights(self, weights: Optional[np.ndarray]) -> "PPFModel":
        """Return a copy voting with ``weights``.

        Separated from training so an ablation can swap weight sources without paying to
        rebuild the table, and so uniform weights provably reduce to the unweighted matcher.
        """
        if weights is None:
            return PPFModel(**{**self.__dict__, "weights": None})
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != self.n_points:
            raise ValueError(f"expected {self.n_points} weights, got {len(w)}")
        if np.any(w < 0):
            raise ValueError("vote weights must be non-negative")
        return PPFModel(**{**self.__dict__, "weights": w})

    # ------------------------------------------------------------------
    def quantise(self, dist: np.ndarray, f2: np.ndarray, f3: np.ndarray,
                 f4: np.ndarray) -> np.ndarray:
        """Pack a batch of features into integer keys. Shared by training and matching."""
        na = self.cfg.n_angle + 1
        step = np.pi / self.cfg.n_angle
        b0 = np.minimum((dist / self.cfg.tau).astype(np.int64), self.n_dist_bins - 1)
        b2 = np.clip((f2 / step).astype(np.int64), 0, self.cfg.n_angle)
        b3 = np.clip((f3 / step).astype(np.int64), 0, self.cfg.n_angle)
        b4 = np.clip((f4 / step).astype(np.int64), 0, self.cfg.n_angle)
        return ((b0 * na + b2) * na + b3) * na + b4

    def lookup(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Half-open ``[lo, hi)`` ranges in the entry arrays for each key.

        The table is fixed once training ends, so re-deriving these bounds by binary search
        on every call is redundant work: two ``searchsorted`` passes over ~2 M sorted keys
        was **42% of total match time** measured on T-LESS obj_000019.  ``key_offsets`` turns
        that into two array reads, 19-35x faster on the same query batches, at ~5-7 MB per
        part built once.

        The binary search is kept as the fallback for parts whose key space is too large to
        materialise (fine angular binning blows it up as ``n_angle^3``), so behaviour is
        identical either way -- only the cost differs.
        """
        return lookup_ranges(self.keys, self.key_offsets, keys)

    # ------------------------------------------------------------------
    @classmethod
    def train(cls, points: np.ndarray, normals: np.ndarray, cfg: PPFConfig,
              spread: str = "default") -> "PPFModel":
        """Enumerate and index every usable ordered model point pair.

        ``spread`` is ``"default"`` (9 neighbours), ``"full"`` (81, Hinterstoisser) or
        ``"none"``.  ``cfg.spread_angle_bins=False`` also disables it, so the ablation can
        attribute credit to spreading independently of the other toggles.
        """
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        nrm = np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        M = len(pts)
        if M < 2:
            raise ValueError("a PPF model needs at least two points")

        frames = frames_to_x(nrm)
        n_dist_bins = int(np.ceil(cfg.max_pair_dist / cfg.tau)) + 1

        model = cls(points=pts, normals=nrm, frames=frames, cfg=cfg,
                    keys=np.empty(0, np.int64), entry_point=np.empty(0, np.int32),
                    entry_point2=np.empty(0, np.int32), entry_alpha=np.empty(0, np.float32),
                    n_dist_bins=n_dist_bins, n_pairs=0)

        ii, jj = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]

        dist, f2, f3, f4 = pair_features(pts, nrm, ii, jj)
        usable = _usable_pairs(dist, nrm, ii, jj, cfg)
        ii, jj = ii[usable], jj[usable]
        dist, f2, f3, f4 = dist[usable], f2[usable], f3[usable], f4[usable]
        if len(ii) == 0:
            raise ValueError(
                "no usable model point pairs -- tau is probably larger than the part; "
                f"tau={cfg.tau * 1e3:.2f}mm, diameter={cfg.diameter * 1e3:.2f}mm")

        alpha = alpha_of(frames[ii], pts[ii], pts[jj])
        base = model.quantise(dist, f2, f3, f4)

        offsets = {"none": None, "default": _SPREAD_9, "full": _SPREAD_81}[spread]
        if not cfg.spread_angle_bins:
            offsets = None

        if offsets is None:
            keys, ep, ep2, ea = base, ii, jj, alpha
        else:
            na = cfg.n_angle + 1
            # Same packing as `quantise`, shifted per offset. Offsets that would leave the
            # valid bin range are dropped rather than wrapped: wrapping would alias the
            # largest distance bin onto the smallest, matching far pairs to near ones.
            b0 = np.minimum((dist / cfg.tau).astype(np.int64), n_dist_bins - 1)
            step = np.pi / cfg.n_angle
            b = [b0,
                 np.clip((f2 / step).astype(np.int64), 0, cfg.n_angle),
                 np.clip((f3 / step).astype(np.int64), 0, cfg.n_angle),
                 np.clip((f4 / step).astype(np.int64), 0, cfg.n_angle)]
            lim = [n_dist_bins - 1, cfg.n_angle, cfg.n_angle, cfg.n_angle]
            ks, eps, eps2, eas = [], [], [], []
            for off in offsets:
                sb = [b[k] + off[k] for k in range(4)]
                ok = np.ones(len(base), bool)
                for k in range(4):
                    ok &= (sb[k] >= 0) & (sb[k] <= lim[k])
                if not ok.any():
                    continue
                ks.append(((sb[0][ok] * na + sb[1][ok]) * na + sb[2][ok]) * na + sb[3][ok])
                eps.append(ii[ok])
                eps2.append(jj[ok])
                eas.append(alpha[ok])
            keys = np.concatenate(ks)
            ep = np.concatenate(eps)
            ep2 = np.concatenate(eps2)
            ea = np.concatenate(eas)

        order = np.argsort(keys, kind="stable")
        keys, ep, ep2, ea = keys[order], ep[order], ep2[order], ea[order]

        cap = _bucket_cap_mask(keys, cfg.max_bucket_entries)
        if cap is not None:
            keys, ep, ep2, ea = keys[cap], ep[cap], ep2[cap], ea[cap]

        model.keys = np.ascontiguousarray(keys)
        model.entry_point = np.ascontiguousarray(ep.astype(np.int32))
        model.entry_point2 = np.ascontiguousarray(ep2.astype(np.int32))
        model.entry_alpha = np.ascontiguousarray(ea.astype(np.float32))
        model.n_pairs = int(len(ii))
        model.key_offsets = _build_key_offsets(keys, n_dist_bins, cfg.n_angle)
        model.point_tree = cKDTree(pts)
        return model


def lookup_ranges(sorted_keys, key_offsets, query):
    """Half-open ``[lo, hi)`` ranges for ``query`` -- the one implementation, host or device.

    Split out of :meth:`PPFModel.lookup` so the GPU path can pass its own device-resident
    arrays through exactly the same logic instead of keeping a second copy that could drift.
    """
    if key_offsets is not None:
        return key_offsets[query], key_offsets[query + 1]
    return (np.searchsorted(sorted_keys, query, side="left"),
            np.searchsorted(sorted_keys, query, side="right"))


# Ceiling on the direct index, in elements. The key space grows as `n_angle^3`, so a part
# configured with fine angular binning can ask for an array far larger than the table it
# indexes -- at n_angle=180 it reaches ~77 M entries (616 MB) to index ~2 M pairs. Every part
# measured so far needs 0.6-0.9 M, so this leaves ~10x headroom before the fallback engages.
MAX_KEY_INDEX = 8_000_000


def _build_key_offsets(sorted_keys: np.ndarray, n_dist_bins: int, n_angle: int
                       ) -> Optional[np.ndarray]:
    """CSR offsets over the full key space, so ``lookup`` is indexing rather than searching.

    ``quantise`` packs four clipped bin indices into ``(((b0*na + b2)*na + b3)*na + b4)`` with
    ``na = n_angle + 1``, so every reachable key lies in ``[0, n_dist_bins * na**3)``.  That
    makes the key space a dense integer range, and the start of each bin can simply be
    tabulated once instead of being binary-searched on every query.

    Returns ``None`` when the key space exceeds :data:`MAX_KEY_INDEX`; the caller then keeps
    using ``searchsorted``, which is slower but produces identical bounds.

    ``sorted_keys`` must already be sorted -- the offsets are positions *in the entry arrays*,
    which is only meaningful if entries are grouped by key. Training sorts before capping and
    the cap preserves order, so this holds at the call site.
    """
    na = int(n_angle) + 1
    keyspace = int(n_dist_bins) * na ** 3
    if keyspace <= 0 or keyspace > MAX_KEY_INDEX or len(sorted_keys) == 0:
        return None
    counts = np.bincount(sorted_keys, minlength=keyspace)
    offsets = np.empty(keyspace + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _bucket_cap_mask(sorted_keys: np.ndarray, cap: int) -> Optional[np.ndarray]:
    """Keep at most ``cap`` entries per feature bin, evenly strided within each bin.

    Needed because feature-space collapse is real and severe: a 100x30x20 box has six
    distinct surface normals, so its bins reach 12 000 entries and matching one instance
    against it would expand to ~1.8e9 votes.

    Strided rather than truncated.  Entries within a bin arrive grouped by model point (the
    pair enumeration is ordered), so keeping the *first* ``cap`` would retain only the
    lowest-indexed model points -- one contiguous patch of the part -- and every pose voted
    from that bin would be biased toward it.  Striding keeps the survivors spread across the
    model points and alpha values the bin actually contains.
    """
    if cap <= 0 or len(sorted_keys) == 0:
        return None
    first = np.searchsorted(sorted_keys, sorted_keys, side="left")
    count = np.searchsorted(sorted_keys, sorted_keys, side="right") - first
    if count.max() <= cap:
        return None
    rank = np.arange(len(sorted_keys)) - first
    stride = np.maximum(1, np.ceil(count / cap).astype(np.int64))
    return (rank % stride) == 0


# ----------------------------------------------------------------------
def pair_features(pts: np.ndarray, nrm: np.ndarray,
                  ii: np.ndarray, jj: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drost's four-number feature for pairs ``(ii, jj)``: ``(|d|, <n_i,d>, <n_j,d>, <n_i,n_j>)``.

    Rotation- and translation-invariant by construction, which is what lets one matched pair
    determine a full pose.
    """
    d = pts[jj] - pts[ii]
    dist = np.linalg.norm(d, axis=1)
    u = d / np.maximum(dist, 1e-12)[:, None]
    dot = lambda a, b: np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    return (dist,
            np.arccos(dot(nrm[ii], u)),
            np.arccos(dot(nrm[jj], u)),
            np.arccos(dot(nrm[ii], nrm[jj])))


def _usable_pairs(dist: np.ndarray, nrm: np.ndarray, ii: np.ndarray, jj: np.ndarray,
                  cfg: PPFConfig) -> np.ndarray:
    """Which pairs carry pose information.

    Pairs shorter than ``min_pair_dist`` are dropped: neighbouring points on a smooth
    surface have near-identical normals, so their feature is near-constant and they flood
    the accumulator with votes that constrain nothing.

    But Hinterstoisser (ECCV 2016) observed that the short pairs whose normals *do* diverge
    are the opposite -- they straddle an edge or a crease and are among the most
    discriminative pairs on the part.  Those are re-admitted.
    """
    ok = dist <= cfg.max_pair_dist
    long_enough = dist >= cfg.min_pair_dist
    if cfg.steep_pair_readmit_deg > 0:
        cos_lim = np.cos(np.deg2rad(cfg.steep_pair_readmit_deg))
        steep = np.einsum("ij,ij->i", nrm[ii], nrm[jj]) < cos_lim
        long_enough |= steep
    return ok & long_enough & (dist > 1e-12)
