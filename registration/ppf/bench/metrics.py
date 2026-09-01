"""Symmetry-aware pose-error metrics, implemented directly rather than via ``bop_toolkit``.

Everything here is in **metres**.  BOP ships its models and its ``models_info.json`` in
millimetres, so ``symmetry_transforms_from_bop`` rescales the translation part of every
symmetry transform on the way in -- a symmetry whose offset is expressed in mm applied to a
model in metres is off by 1000x and silently turns a correct pose into a gross failure.

Why not just import ``bop_toolkit_lib``
    Because this package's contract is that it depends on NumPy, SciPy and Open3D and
    nothing else, so it can be lifted into another project unchanged.  MSSD, ADD and ADI are
    each three lines; taking a dependency for them would cost more than writing them.
    ``registration/tests/test_ppf.py`` cross-checks this implementation against
    ``bop_toolkit_lib`` wherever that package happens to be installed, so "written here"
    does not mean "unverified".

Why not compare rotations against a single ground-truth pose
    Because for a symmetric part several rotations name the same physical placement, so that
    comparison fails ~75% of the time on a cuboid for reasons that say nothing about the
    matcher.  Switching angular scoring off entirely is the other common dodge, and it cannot
    tell "correctly oriented modulo symmetry" from "completely wrong" -- which for bin
    picking is the difference that ruins the gripper approach vector.

    So MSSD is the primary metric: it minimises surface distance over the symmetry group, so
    it is symmetry-aware by construction and needs no angular threshold at all.  Measured on
    T-LESS obj_000001 (continuous symmetry about z): a 37 degree rotation about z scores
    MSSD 0.13 mm -- essentially free, correctly -- while ADD charges 7.65 mm for it.  About
    x, which is not a symmetry, MSSD charges 20.9 mm.

Quotienting happens over **global** symmetry only.  A view-dependent ambiguity is not a
symmetry: the part genuinely is not invariant and the returned pose is genuinely wrong, so
folding those into the quotient would hide real failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["PoseError", "symmetry_transforms_from_bop", "evaluate_pose", "summarise",
           "mssd", "add", "adi", "TIGHT", "LOOSE"]

# Accuracy gates in metres/degrees. TIGHT is the coarse+fine target of the industrial
# pipeline this was built for; LOOSE is its coarse gate.
TIGHT = (0.002, 5.0)
LOOSE = (0.005, 10.0)


@dataclass
class PoseError:
    """One estimate scored against one ground-truth pose."""

    mssd: float                      # m, minimised over the global symmetry group
    add: float                       # m, symmetry-agnostic (kept as the naive baseline)
    adi: float                       # m, closest-point; blind to symmetry AND to real flips
    te: float                        # m, translation error
    re_deg: float                    # deg, raw rotation error against the single GT
    re_sym_deg: float                # deg, minimised over the global symmetry group
    diameter: float = 0.0            # m, of the model -- for the BOP-relative gate

    def passes(self, gate=TIGHT) -> bool:
        """Recall gate on the *symmetry-aware* pair, not the raw one."""
        return self.te < gate[0] and self.re_sym_deg < gate[1]

    def passes_bop(self, theta: float = 0.2) -> bool:
        """BOP's MSSD criterion: correct iff ``MSSD < theta * diameter``.

        This is the gate a *coarse* matcher should be judged on, and the one that makes the
        numbers comparable to published work.  ``TIGHT`` (2 mm / 5 deg) is the target for a
        coarse **plus fine** pipeline; PPF alone returns the best pose on a quantised
        accumulator with no refinement, so scoring it there judges it for missing a step it
        does not contain.  Both are reported: TIGHT says "ready to grasp without refinement",
        this one says "the coarse stage found the object".
        """
        return self.diameter > 0 and self.mssd < theta * self.diameter


# ----------------------------------------------------------------------
# The three surface-distance metrics
# ----------------------------------------------------------------------

def _apply(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64).reshape(3)


def add(R_est, t_est, R_gt, t_gt, pts: np.ndarray) -> float:
    """Average Distance of corresponding model points. Symmetry-agnostic by design."""
    return float(np.linalg.norm(_apply(R_est, t_est, pts) - _apply(R_gt, t_gt, pts),
                                axis=1).mean())


def adi(R_est, t_est, R_gt, t_gt, pts: np.ndarray) -> float:
    """ADD using the *nearest* model point instead of the corresponding one.

    Blind to symmetry, which is the point -- and also blind to genuine flips, which is why it
    is reported but never used as the gate.

    Direction matters and is not arbitrary: distances run **from the ground-truth vertices to
    the nearest estimated ones** (Hinterstoisser ACCV'12, and what ``bop_toolkit`` computes).
    Nearest-neighbour is asymmetric, so querying the other way round is a different number --
    measured 29.4 mm against 25.2 mm on the same pose here.
    """
    from scipy.spatial import cKDTree

    return float(cKDTree(_apply(R_est, t_est, pts))
                 .query(_apply(R_gt, t_gt, pts), k=1)[0].mean())


def mssd(R_est, t_est, R_gt, t_gt, pts: np.ndarray,
         syms: Sequence[Dict[str, np.ndarray]]) -> float:
    """Maximum Symmetry-aware Surface Distance (BOP).

    ``min`` over the symmetry group of the ``max`` vertex displacement.  The max (not the
    mean) is what makes it a *worst-case* surface agreement, so a pose cannot pass by being
    right about most of the part.
    """
    est = _apply(R_est, t_est, pts)
    best = np.inf
    for s in (syms or [{"R": np.eye(3), "t": np.zeros((3, 1))}]):
        # The symmetry acts in the model frame, before the ground-truth placement.
        sym_pts = pts @ np.asarray(s["R"], dtype=np.float64).T \
            + np.asarray(s["t"], dtype=np.float64).reshape(3)
        d = np.linalg.norm(est - _apply(R_gt, t_gt, sym_pts), axis=1).max()
        best = min(best, float(d))
    return float(best)


# ----------------------------------------------------------------------
# Symmetry groups
# ----------------------------------------------------------------------

def symmetry_transforms_from_bop(model_info: dict, diameter_m: float = 0.0,
                                 max_disc_step: float = 0.01,
                                 units_to_m: float = 1e-3) -> List[Dict[str, np.ndarray]]:
    """Global symmetry group from a BOP ``models_info.json`` entry, converted to metres.

    Reproduces ``bop_toolkit_lib.misc.get_symmetry_transformations`` exactly -- verified
    against it element by element in ``registration/tests/test_ppf.py``.  Three details of
    that function are easy to get wrong, and each one silently changes the group:

    * ``max_disc_step`` is **a fraction of the object diameter, not an angle**.  It bounds
      how far the vertex furthest from the axis may travel between consecutive discretised
      rotations, which is why the step count is ``ceil(pi / max_disc_step)`` (from
      ``pi * diam / (max_disc_step * diam)``) rather than ``ceil(2*pi / step)``.
    * The discretised continuous set **includes the identity rotation** (``i`` runs from 0),
      because the composition below relies on it to carry the bare discrete symmetries.
    * When a continuous axis exists, the output is *only* the composed transforms -- the
      unmodified discrete ones are not appended as well.  Appending them duplicates every
      discrete symmetry, since the ``i = 0`` continuous element already reproduces it.

    ``units_to_m`` rescales the *translation* part only.  BOP's offsets are in millimetres;
    every length in this package is metres, and applying a mm offset to a metre-scale model
    is a 1000x error that reads as a gross pose failure rather than as a unit bug.
    """
    from scipy.spatial.transform import Rotation as Rot

    disc: List[Dict[str, np.ndarray]] = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    for sym in (model_info or {}).get("symmetries_discrete", []):
        m = np.asarray(sym, dtype=np.float64).reshape(4, 4)
        disc.append({"R": m[:3, :3].copy(),
                     "t": m[:3, 3].reshape(3, 1) * units_to_m})

    cont: List[Dict[str, np.ndarray]] = []
    for sym in (model_info or {}).get("symmetries_continuous", []):
        axis = np.asarray(sym["axis"], dtype=np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        offset = np.asarray(sym["offset"], dtype=np.float64).reshape(3, 1) * units_to_m
        steps = int(np.ceil(np.pi / max_disc_step))
        step = 2.0 * np.pi / steps
        for i in range(0, steps):
            R = Rot.from_rotvec(axis * (i * step)).as_matrix()
            # Rotating about an axis that misses the origin carries a translation.
            cont.append({"R": R, "t": -R @ offset + offset})

    out: List[Dict[str, np.ndarray]] = []
    for d in disc:
        if cont:
            for c in cont:
                out.append({"R": c["R"] @ d["R"], "t": c["R"] @ d["t"] + c["t"]})
        else:
            out.append(d)
    return out


def _sym_rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray,
                            syms: Sequence[Dict[str, np.ndarray]]) -> float:
    best = 180.0
    for s in syms:
        Rg = R_gt @ s["R"]
        tr = float(np.trace(Rg.T @ R_est))
        best = min(best, float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))))
    return best


def _rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    tr = float(np.trace(np.asarray(R_gt).T @ np.asarray(R_est)))
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


# ----------------------------------------------------------------------
def evaluate_pose(T_est: np.ndarray, T_gt: np.ndarray, model_points: np.ndarray,
                  syms: Sequence[Dict[str, np.ndarray]],
                  diameter: float = 0.0) -> PoseError:
    """Score one estimated pose. ``model_points`` and both transforms are in metres."""
    T_est = np.asarray(T_est, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    R_e, t_e = T_est[:3, :3], T_est[:3, 3]
    R_g, t_g = T_gt[:3, :3], T_gt[:3, 3]
    pts = np.asarray(model_points, dtype=np.float64).reshape(-1, 3)
    syms = list(syms) or [{"R": np.eye(3), "t": np.zeros((3, 1))}]

    return PoseError(
        mssd=mssd(R_e, t_e, R_g, t_g, pts, syms),
        add=add(R_e, t_e, R_g, t_g, pts),
        adi=adi(R_e, t_e, R_g, t_g, pts),
        te=float(np.linalg.norm(t_e - t_g)),
        re_deg=_rotation_error_deg(R_e, R_g),
        re_sym_deg=_sym_rotation_error_deg(R_e, R_g, syms),
        diameter=float(diameter),
    )


def summarise(errors: Sequence[PoseError], n_expected: Optional[int] = None,
              theta: float = 0.2) -> Dict[str, float]:
    """Aggregate per-instance errors, with the confidence interval attached.

    The CI is not decoration.  At 26 instances its half-width on a ~0.5 recall is +/- 19
    points, wider than most differences anyone will want to read off this table -- so a
    summary that reports recall without it invites reading noise as a result.
    """
    n = len(errors)
    if n == 0:
        return {"n": 0}
    total = n_expected or n
    bop = float(np.mean([e.passes_bop(theta) for e in errors]))
    tight = float(np.mean([e.passes(TIGHT) for e in errors]))
    loose = float(np.mean([e.passes(LOOSE) for e in errors]))
    return {
        "n": n,
        "found": n / total,
        "recall_bop": bop,
        "recall_tight": tight,
        "recall_loose": loose,
        "ci95_half": 1.96 * float(np.sqrt(max(bop * (1 - bop), 1e-9) / total)),
        "mssd_p50": float(np.median([e.mssd for e in errors])),
        "add_p50": float(np.median([e.add for e in errors])),
        "te_p50": float(np.median([e.te for e in errors])),
        "re_sym_p50": float(np.median([e.re_sym_deg for e in errors])),
    }
