"""Pose-error metrics for the ablation.

Everything here is in **metres**, matching the rest of the pipeline. BOP ships its models in
millimetres, so ``symmetry_transforms_from_bop`` rescales the translation part of every
symmetry transform on the way in -- a symmetry whose offset is expressed in mm applied to a
model in metres is off by 1000x and silently turns a correct pose into a gross failure.

Why not just compare rotations against a single ground-truth pose
    Because for a symmetric part several rotations name the same physical placement, so that
    comparison fails ~75% of the time on a cuboid for reasons that say nothing about the
    matcher. The existing MechVision-side evaluator dodges this by switching angular scoring
    off entirely (``ANG_THRESH_REGIME_GATE = 360``), which is safe but cannot tell "correctly
    oriented modulo symmetry" from "completely wrong" -- and for bin picking the gripper
    approach vector is exactly what a wrong orientation ruins.

    So MSSD is the primary metric: it minimises surface distance over the symmetry group, so
    it is symmetry-aware by construction and needs no angular threshold at all. Measured on
    T-LESS obj_000001 (continuous symmetry about z): a 37 degree rotation about z scores
    MSSD 0.13 mm -- essentially free, correctly -- while ADD charges 7.65 mm for it. About x,
    which is not a symmetry, MSSD charges 20.9 mm.

The distinction that matters most, and that BOP alone cannot make
    Quotienting happens over **global** symmetry only. A view-dependent ambiguity is not a
    symmetry: the part genuinely is not invariant, the returned pose is genuinely wrong, and
    it will fail at the gripper. Folding those into the quotient would hide precisely the
    failures this project exists to fix. They are instead counted as failures and *tagged*
    against the ``geometry.ambiguity`` axis that predicted them -- and that tagged count is a
    headline result, because it measures whether the heat map is telling the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["PoseError", "symmetry_transforms_from_bop", "symmetry_transforms_from_profile",
           "evaluate_pose", "summarise", "TIGHT", "LOOSE"]

# The repo's existing accuracy gates (MM_Optimizer/search_config.py), in metres/degrees.
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
    axis_err_deg: float = float("nan")   # deg between predicted and GT symmetry axis
    phase_err_deg: float = float("nan")  # deg about that axis, mod 360/fold; NaN if continuous
    ambiguity_tagged: bool = False   # a view-dependent axis explains this failure
    diameter: float = 0.0            # m, of the model -- for the BOP-relative gate

    def passes(self, gate=TIGHT) -> bool:
        """Recall gate on the *symmetry-aware* pair, not the raw one."""
        return self.te < gate[0] and self.re_sym_deg < gate[1]

    def passes_bop(self, theta: float = 0.2) -> bool:
        """BOP's MSSD criterion: correct iff ``MSSD < theta * diameter``.

        This is the gate a *coarse* matcher should be judged on, and it is the one that makes
        the numbers comparable to published work.

        ``TIGHT`` (2 mm / 5 deg) comes from ``MM_Optimizer/search_config.py``, where it is
        the target for MechVision's coarse **plus fine** pipeline. PPF alone is a coarse
        stage -- it returns the best pose on a quantised accumulator, with no refinement -- so
        scoring it there judges it for missing a step it does not contain. Worse for the
        ablation, it compresses every arm toward zero and hides the differences between them:
        on T-LESS obj_000018 (111 mm diameter) the median MSSD was 15.4 mm, which is a
        comfortable pass at BOP's 22 mm and a clear fail at 2 mm.

        Both gates are reported. TIGHT says "ready to grasp without refinement"; this one
        says "the coarse stage found the object", which is the question the arms differ on.
        """
        return self.diameter > 0 and self.mssd < theta * self.diameter


def _as_rt(T: np.ndarray):
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3], T[:3, 3].reshape(3, 1)


def symmetry_transforms_from_bop(model_info: dict, diameter_m: float,
                                 max_disc_step: float = 0.01) -> List[Dict[str, np.ndarray]]:
    """Global symmetry group from a BOP ``models_info.json`` entry, converted to metres.

    Continuous symmetries are discretised the way BOP does it -- finely enough that the
    furthest model vertex moves less than ``max_disc_step`` of the diameter between steps.
    """
    from bop_toolkit_lib import misc

    syms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=max_disc_step)
    out = []
    for s in syms:
        out.append({"R": np.asarray(s["R"], dtype=np.float64),
                    # models_info is in mm; every length in this repo is metres.
                    "t": np.asarray(s["t"], dtype=np.float64).reshape(3, 1) * 1e-3})
    return out


def symmetry_transforms_from_profile(profile, max_disc_step_deg: float = 5.0
                                     ) -> List[Dict[str, np.ndarray]]:
    """Global symmetry group from an ``AmbiguityProfile``, for parts BOP has never seen.

    **Only ``is_global`` axes are used.** A view-dependent axis maps the *visible patch*
    somewhere else, not the model onto itself, so treating it as a symmetry would forgive a
    pose the gripper will miss.
    """
    from scipy.spatial.transform import Rotation as Rot

    out = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    if profile is None:
        return out
    for ax in getattr(profile, "axes", []):
        if not getattr(ax, "is_global", False):
            continue
        d = np.asarray(ax.direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        p = np.asarray(ax.point, dtype=np.float64).reshape(3, 1)
        fold = int(getattr(ax, "fold", 1))
        angles = (np.arange(max_disc_step_deg, 360.0, max_disc_step_deg) if fold == 0
                  else [360.0 * k / fold for k in range(1, max(fold, 1))])
        for a in angles:
            R = Rot.from_rotvec(d * np.deg2rad(a)).as_matrix()
            # Rotating about an axis that misses the origin carries a translation.
            out.append({"R": R, "t": p - R @ p})
    return out


def _sym_rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray,
                            syms: Sequence[Dict[str, np.ndarray]]) -> float:
    best = 180.0
    for s in syms:
        Rg = R_gt @ s["R"]
        tr = float(np.trace(Rg.T @ R_est))
        best = min(best, float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))))
    return best


def _axis_phase(R_est: np.ndarray, R_gt: np.ndarray, profile) -> tuple:
    """Split rotation error into axis-direction error and phase about that axis.

    Reported separately because one scalar cannot distinguish a perfect axis with a 180
    degree flip (a C2 confusion) from an axis that is off by 30 degrees (a plain miss), and
    that difference is much of what the ablation is trying to measure.

    Phase is ``NaN`` for a continuous axis, where it is genuinely meaningless -- recorded as
    NaN rather than zero so it cannot be averaged into a summary as if it were a measurement.
    """
    axes = [a for a in getattr(profile, "axes", []) if getattr(a, "is_global", False)] \
        if profile is not None else []
    if not axes:
        return float("nan"), float("nan")
    ax = axes[0]
    d = np.asarray(ax.direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)

    d_gt, d_est = R_gt @ d, R_est @ d
    cos = float(np.clip(abs(np.dot(d_gt, d_est)), -1.0, 1.0))   # abs: +d and -d are one axis
    axis_err = float(np.degrees(np.arccos(cos)))

    fold = int(getattr(ax, "fold", 1))
    if fold == 0:
        return axis_err, float("nan")
    # Residual rotation once the axis itself is aligned.
    tr = float(np.trace(R_gt.T @ R_est))
    total = float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))
    step = 360.0 / max(fold, 1)
    return axis_err, float(min(total % step, step - (total % step)))


def evaluate_pose(T_est: np.ndarray, T_gt: np.ndarray, model_points: np.ndarray,
                  syms: Sequence[Dict[str, np.ndarray]],
                  profile=None, diameter: float = 0.0) -> PoseError:
    """Score one estimated pose. ``model_points`` and both transforms are in metres."""
    from bop_toolkit_lib import pose_error

    R_e, t_e = _as_rt(T_est)
    R_g, t_g = _as_rt(T_gt)
    pts = np.asarray(model_points, dtype=np.float64)
    syms = list(syms) or [{"R": np.eye(3), "t": np.zeros((3, 1))}]

    err = PoseError(
        mssd=float(pose_error.mssd(R_e, t_e, R_g, t_g, pts, syms)),
        add=float(pose_error.add(R_e, t_e, R_g, t_g, pts)),
        adi=float(pose_error.adi(R_e, t_e, R_g, t_g, pts)),
        te=float(np.linalg.norm(t_e - t_g)),
        re_deg=float(pose_error.re(R_e, R_g)),
        re_sym_deg=_sym_rotation_error_deg(R_e, R_g, syms),
        diameter=float(diameter),
    )
    err.axis_err_deg, err.phase_err_deg = _axis_phase(R_e, R_g, profile)
    if not err.passes(LOOSE):
        err.ambiguity_tagged = _explained_by_view_ambiguity(R_e, R_g, profile)
    return err


def _explained_by_view_ambiguity(R_est: np.ndarray, R_gt: np.ndarray, profile,
                                 tol_deg: float = 12.0) -> bool:
    """Is this failure a flip about a *view-dependent* ambiguity axis?

    Answering yes does not excuse the pose -- it is still counted as a failure. It records
    that the heat map predicted this specific way of being wrong, which is the cheapest
    available test of whether the ambiguity analysis is describing reality.
    """
    from scipy.spatial.transform import Rotation as Rot

    if profile is None:
        return False
    for ax in getattr(profile, "axes", []):
        if getattr(ax, "is_global", False):
            continue
        d = np.asarray(ax.direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        fold = int(getattr(ax, "fold", 1))
        angles = (np.arange(15.0, 360.0, 15.0) if fold == 0
                  else [360.0 * k / fold for k in range(1, max(fold, 2))])
        for a in angles:
            R = Rot.from_rotvec(d * np.deg2rad(a)).as_matrix()
            tr = float(np.trace((R_gt @ R).T @ R_est))
            if np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))) < tol_deg:
                return True
    return False


def summarise(errors: Sequence[PoseError], n_expected: Optional[int] = None) -> Dict[str, float]:
    """Aggregate per-instance errors, with the confidence interval attached.

    The CI is not decoration. At 26 instances its half-width on a ~0.5 recall is +/- 19
    points, which is wider than any arm difference measured so far -- so a summary that
    reports recall without it invites reading noise as a result.
    """
    n = len(errors)
    if n == 0:
        return {"n": 0}
    total = n_expected or n
    mssd = np.array([e.mssd for e in errors])
    te = np.array([e.te for e in errors])
    tight = float(np.mean([e.passes(TIGHT) for e in errors]))
    loose = float(np.mean([e.passes(LOOSE) for e in errors]))
    half = 1.96 * float(np.sqrt(max(tight * (1 - tight), 1e-9) / total))
    return {
        "n": n,
        "found": n / total,
        "recall_tight": tight,
        "recall_loose": loose,
        "ci95_half": half,
        "mssd_p50": float(np.median(mssd)),
        "te_p50": float(np.median(te)),
        "re_sym_p50": float(np.median([e.re_sym_deg for e in errors])),
        "ambiguity_flips": float(np.mean([e.ambiguity_tagged for e in errors])),
    }
