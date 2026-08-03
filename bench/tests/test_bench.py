"""Tests for the benchmark harness — arm construction, metrics, and dataset conventions.

Run:  python -m pytest bench/tests/test_bench.py -q

The harness decides what the ablation *measures*, so a silent fault here does not crash
anything — it produces a confident wrong answer about which method is better. Two of these
tests exist because exactly that happened during development:

* ``test_knee_mask_survives_an_atom_at_zero`` — 41.5% of a real heat map is exactly 0.0, the
  CDF knee landed on that spike, the threshold came back 0.0, and the "pruned" arm kept 100%
  of the points. It was a byte-identical copy of the baseline and would have been reported as
  evidence that pruning is harmless.
* ``test_symmetry_aware_error_forgives_only_real_symmetry`` — the whole point of the metric
  is to separate "correct modulo symmetry" from "wrong". Quotienting too much silently
  passes real failures; too little fails correct poses on symmetric parts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation as Rot

from bench.arms import ARMS, ArmContext, _knee_mask, build_arm
from bench.metrics import (LOOSE, TIGHT, PoseError, evaluate_pose,
                           symmetry_transforms_from_profile)
from geometry.ambiguity import AmbiguityAxis, AmbiguityProfile
from registration.ppf import PPFConfig, downsample


# ─────────────────────────────────────────────────────────────────────────────
# Arm construction
# ─────────────────────────────────────────────────────────────────────────────

def test_knee_mask_survives_an_atom_at_zero():
    """A large spike at the minimum must not swallow the knee.

    The ambiguity heat map is not smooth: points explained exactly by some ambiguity
    transform score a hard 0.0, and on a real part that is ~40% of them. That spike is a
    vertical jump at the left edge of the CDF and therefore the furthest point from the
    chord, so a naive knee returns the minimum itself and ``v >= threshold`` selects
    everything.
    """
    rng = np.random.default_rng(0)
    heat = np.concatenate([np.zeros(4000), rng.uniform(0.05, 1.0, 6000)])

    mask = _knee_mask(heat)
    assert not mask.all(), "the atom at zero swallowed the knee — nothing was pruned"
    assert not mask[:4000].any(), "zero-scoring points must be dropped, they are the point"
    assert mask.sum() >= 50


def test_knee_mask_keeps_everything_when_there_is_nothing_to_split():
    """A part with no recovered ambiguity axes scores 1.0 everywhere (measured: the Stanford
    bunny). There is no upper tail to keep, so the arm must degrade to the baseline rather
    than emit an empty or arbitrary model."""
    assert _knee_mask(np.ones(1000)).all()
    assert _knee_mask(np.full(1000, 0.03)).all()


def test_arms_produce_distinct_models_that_still_span_the_part():
    """Each cloud-modifying arm must actually change the cloud, and must not collapse it
    into one clustered patch — a model whose points share a small region has no lever arm on
    rotation and fails for a reason unrelated to what the arm is testing."""
    mesh = o3d.geometry.TriangleMesh.create_torus(0.03, 0.01, 60, 30)
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(0)
    pcd = mesh.sample_points_uniformly(8000)
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)

    cfg = PPFConfig.derive(pts, model_target_points=300)
    rng = np.random.default_rng(0)
    heat = rng.uniform(0.0, 1.0, len(pts))
    ctx = ArmContext(points=pts, normals=nrm, tau=cfg.tau, heat=heat)

    full_extent = np.linalg.norm(pts.max(0) - pts.min(0))
    sizes = {}
    for name in ARMS:
        a_pts, a_nrm, _ = build_arm(name, ctx)
        assert len(a_pts) == len(a_nrm)
        sizes[name] = len(a_pts)
        ext = np.linalg.norm(a_pts.max(0) - a_pts.min(0))
        assert ext > 0.7 * full_extent, f"{name} collapsed the model into a patch"

    assert sizes["B_heat_prune"] < sizes["A_uniform"], "pruning arm did not prune"
    assert sizes["A_uniform"] == len(pts)


def test_weight_arms_do_not_touch_the_cloud():
    """E/F/G must reuse the baseline's exact table. If they changed the cloud too, a measured
    difference could not be attributed to the weights."""
    mesh = o3d.geometry.TriangleMesh.create_box(0.06, 0.04, 0.02)
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(0)
    pcd = mesh.sample_points_uniformly(4000)
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    cfg = PPFConfig.derive(pts, model_target_points=200)
    ctx = ArmContext(points=pts, normals=nrm, tau=cfg.tau,
                     heat=np.random.default_rng(0).uniform(0, 1, len(pts)))

    for name in ("E_heat_weight", "F_ppf_weight", "G_combined"):
        a_pts, _, _ = build_arm(name, ctx)
        assert len(a_pts) == len(pts), f"{name} modified the cloud"
        assert not ARMS[name].retrains


def test_heat_needing_arms_refuse_to_run_without_a_heat_map():
    """Better to skip an arm than to invent a weighting. Parts with no recovered ambiguity
    axes have no heat map at all, and a fabricated one would be reported as a real result."""
    pts = np.random.default_rng(0).normal(size=(500, 3))
    nrm = np.tile([0.0, 0.0, 1.0], (500, 1))
    ctx = ArmContext(points=pts, normals=nrm, tau=0.01, heat=None)
    for name, spec in ARMS.items():
        if spec.needs_heat:
            with pytest.raises(ValueError):
                build_arm(name, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _cylinder_points(n=3000, r=0.02, h=0.08):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(r, h, resolution=48)
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(0)
    return np.asarray(mesh.sample_points_uniformly(n).points)


def _profile_with_axis(fold: int, is_global: bool = True) -> AmbiguityProfile:
    ax = AmbiguityAxis(direction=np.array([0.0, 0.0, 1.0]), point=np.zeros(3), fold=fold,
                       angles_deg=[], is_global=is_global,
                       view_fraction=1.0, area_fraction=0.99)
    return AmbiguityProfile(axes=[ax], dominant=ax)


def _T(R=np.eye(3), t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_symmetry_aware_error_forgives_only_real_symmetry():
    """Rotating about a genuine symmetry axis must be free; about any other axis must not.

    This is the property that lets the benchmark score symmetric parts at all, instead of
    switching angular scoring off the way the MechVision-side evaluator has to.
    """
    pts = _cylinder_points()
    syms = symmetry_transforms_from_profile(_profile_with_axis(fold=0))

    about_axis = evaluate_pose(_T(Rot.from_euler("z", 37, degrees=True).as_matrix()),
                               _T(), pts, syms, diameter=0.09)
    off_axis = evaluate_pose(_T(Rot.from_euler("x", 37, degrees=True).as_matrix()),
                             _T(), pts, syms, diameter=0.09)

    assert about_axis.re_sym_deg < 10.0, "rotation about the symmetry axis was not forgiven"
    assert about_axis.mssd < 0.2 * 0.09
    assert off_axis.re_sym_deg > 25.0, "a non-symmetry rotation was wrongly forgiven"
    assert off_axis.mssd > about_axis.mssd * 5


def test_view_dependent_axes_are_never_quotiented_away():
    """A view-dependent ambiguity is not a symmetry — the part is genuinely not invariant and
    the pose is genuinely wrong. Forgiving it would hide exactly the failures this project
    exists to fix."""
    view_only = _profile_with_axis(fold=2, is_global=False)
    syms = symmetry_transforms_from_profile(view_only)
    assert len(syms) == 1, "a non-global axis leaked into the symmetry group"
    assert np.allclose(syms[0]["R"], np.eye(3))


def test_bop_gate_is_relative_to_part_size():
    """MSSD < 0.2*diameter, so the same absolute error passes on a big part and fails on a
    small one. A fixed millimetre gate would make the sweep's per-part results incomparable."""
    e = PoseError(mssd=0.010, add=0, adi=0, te=0, re_deg=0, re_sym_deg=0, diameter=0.100)
    assert e.passes_bop()
    e.diameter = 0.040
    assert not e.passes_bop()
    e.diameter = 0.0
    assert not e.passes_bop(), "an unknown diameter must not silently pass"


def test_tight_gate_uses_symmetry_aware_rotation():
    """The gates score ``re_sym_deg``, not the raw angle: on a symmetric part the raw angle
    is meaningless and would fail correct poses."""
    e = PoseError(mssd=0, add=0, adi=0, te=0.001, re_deg=179.0, re_sym_deg=1.0, diameter=0.1)
    assert e.passes(TIGHT) and e.passes(LOOSE)


def test_symmetry_group_from_profile_covers_the_fold():
    """A C4 axis must yield 4 group members, not 4 arbitrary rotations."""
    syms = symmetry_transforms_from_profile(_profile_with_axis(fold=4))
    assert len(syms) == 4
    angles = sorted(round(float(np.degrees(Rot.from_matrix(s["R"]).magnitude())), 1)
                    for s in syms)
    assert angles == [0.0, 90.0, 90.0, 180.0]


def test_offset_axis_symmetry_carries_a_translation():
    """An axis that misses the origin induces a translation as well as a rotation. Dropping it
    would place the rotated model somewhere the part is not."""
    ax = AmbiguityAxis(direction=np.array([0.0, 0.0, 1.0]),
                       point=np.array([0.05, 0.0, 0.0]), fold=2, angles_deg=[],
                       is_global=True, view_fraction=1.0, area_fraction=0.99)
    syms = symmetry_transforms_from_profile(AmbiguityProfile(axes=[ax], dominant=ax))
    non_identity = [s for s in syms if not np.allclose(s["R"], np.eye(3))]
    assert non_identity, "the C2 member is missing"
    assert np.linalg.norm(non_identity[0]["t"]) > 1e-6
