"""Tests for registration.ppf — the from-scratch PPF + Hough matcher.

Run:  python -m pytest registration/tests/test_ppf.py -q

The properties asserted here are the ones that, if they silently broke, would invalidate the
ablation this matcher exists to run:

1. **The frame convention round-trips.**  PPF encodes a pose as (model point, alpha); train,
   match and pose reconstruction must share one convention.  A sign error there still
   produces plausible-looking poses, just wrong ones, so it is asserted numerically rather
   than trusted.
2. **Symmetric parts come back in the right symmetry orbit**, not at one nominated pose.
   Scoring a box against a single ground-truth rotation would fail three quarters of the
   time for reasons that have nothing to do with the matcher.
3. **Uniform weights reproduce the unweighted result exactly.**  Weighted voting is the
   primary hypothesis of the study; if weighting silently changed the answer at w=1 then no
   comparison between weighted and unweighted arms would mean anything.
4. **Derived parameters are scale-invariant.**  A 20 mm part and a 400 mm part must be
   treated the same way, or the "no per-part configuration" claim is empty.

Point counts are kept small so the suite stays quick; production uses larger clouds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation as Rot

from registration.ppf import (
    PPFConfig,
    PPFModel,
    SensorProfile,
    alpha_of,
    downsample,
    frames_to_x,
    match,
    pose_from_correspondence,
)

FAST = dict(model_target_points=220)
N_SAMPLE = 6000


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _centred(mesh):
    mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    mesh.compute_vertex_normals()
    return mesh


def _cloud(mesh, n=N_SAMPLE):
    o3d.utility.random.seed(0)
    pcd = mesh.sample_points_uniformly(n, use_triangle_normal=False)
    return np.asarray(pcd.points), np.asarray(pcd.normals)


def _box():
    return _centred(o3d.geometry.TriangleMesh.create_box(0.100, 0.030, 0.020))


def _bumpy():
    """A box with an off-centre boss — asymmetric, so it has a unique correct pose.

    A bare primitive is a bad correctness fixture precisely because it is symmetric: any of
    several poses is right, so a test on one of them cannot distinguish a working matcher
    from a lucky one.
    """
    box = o3d.geometry.TriangleMesh.create_box(0.080, 0.050, 0.020)
    boss = o3d.geometry.TriangleMesh.create_cylinder(0.008, 0.020, resolution=24)
    boss.translate((0.022, 0.014, 0.020))
    return _centred(box + boss)


def _build(mesh, **kw):
    pts, nrm = _cloud(mesh)
    cfg = PPFConfig.derive(pts, **{**FAST, **kw})
    mp, mn = downsample(pts, nrm, cfg.tau)
    return PPFModel.train(mp, mn, cfg), pts, nrm


def _apply(T, pts, nrm):
    return pts @ T[:3, :3].T + T[:3, 3], nrm @ T[:3, :3].T


def _pose_error(T_est, T_gt):
    E = np.linalg.inv(T_gt) @ T_est
    ang = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    return float(np.linalg.norm(E[:3, 3])), float(ang)


def _orbit_error(T_est, T_gt, sym_rotations):
    """Smallest pose error over a symmetry group — the only meaningful score for a
    symmetric part, since every group element names the same physical placement."""
    best = (1e9, 1e9)
    for S in sym_rotations:
        G = np.eye(4)
        G[:3, :3] = S
        cand = _pose_error(T_est, T_gt @ G)
        if cand[1] < best[1]:
            best = cand
    return best


def _box_c2_group():
    """The three 180-degree face rotations of a cuboid, plus identity."""
    return [np.eye(3)] + [Rot.from_rotvec(np.pi * np.eye(3)[k]).as_matrix() for k in range(3)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. The local-frame convention
# ─────────────────────────────────────────────────────────────────────────────

def test_frames_map_normals_onto_x_including_the_degenerate_ones():
    """The ±x normals have no unique rotation axis, so the generic Rodrigues formula
    divides by zero there. Both are exercised explicitly."""
    rng = np.random.default_rng(0)
    n = rng.normal(size=(200, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n = np.vstack([n, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])

    R = frames_to_x(n)
    assert np.allclose(np.einsum("nij,nj->ni", R, n), [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_alpha_and_pose_reconstruction_are_inverse():
    """``pose_from_correspondence`` must invert exactly what ``alpha_of`` measures.

    This is the single assumption the whole matcher rests on: one matched point pair yields
    the full pose. A sign slip here yields poses that look reasonable and are wrong, which
    no downstream test would attribute to the convention.
    """
    rng = np.random.default_rng(1)
    n = 300
    pm, qm = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    nm = rng.normal(size=(n, 3))
    nm /= np.linalg.norm(nm, axis=1, keepdims=True)

    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=7).as_matrix()
    T[:3, 3] = [0.31, -0.22, 0.74]
    ps, qs = pm @ T[:3, :3].T + T[:3, 3], qm @ T[:3, :3].T + T[:3, 3]
    ns = nm @ T[:3, :3].T

    Rm, Rs = frames_to_x(nm), frames_to_x(ns)
    rec = pose_from_correspondence(pm, Rm, ps, Rs,
                                   alpha_of(Rm, pm, qm) - alpha_of(Rs, ps, qs))
    assert np.allclose(rec, T, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recovery
# ─────────────────────────────────────────────────────────────────────────────

def test_identity_recovery_on_an_asymmetric_part():
    model, pts, nrm = _build(_bumpy())
    best = match(model, pts, nrm).best
    assert best is not None
    pos, ang = _pose_error(best.T, np.eye(4))
    assert pos < 2e-3, f"translation off by {pos * 1e3:.2f} mm"
    assert ang < 5.0, f"rotation off by {ang:.2f} deg"
    assert best.score > 0.9


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_known_transform_recovery_on_an_asymmetric_part(seed):
    """The core claim: an arbitrary rigid placement is recovered to coarse-match tolerance.

    5 mm / 10 deg is the repo's LOOSE gate (``MM_Optimizer/search_config.py``). Coarse PPF
    is not expected to reach the 2 mm / 5 deg TIGHT gate unaided — that is what a fine
    refinement stage is for — so asserting the tight gate here would encode a false
    expectation of the algorithm.
    """
    model, pts, nrm = _build(_bumpy())
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=seed).as_matrix()
    T[:3, 3] = np.random.default_rng(seed).uniform(-0.05, 0.05, 3)

    best = match(model, *_apply(T, pts, nrm)).best
    assert best is not None
    pos, ang = _pose_error(best.T, T)
    assert pos < 5e-3, f"translation off by {pos * 1e3:.2f} mm"
    assert ang < 10.0, f"rotation off by {ang:.2f} deg"


def test_symmetric_part_lands_in_the_symmetry_orbit():
    """A cuboid's C2 axes make four rotations physically identical.

    Scored against one nominated ground truth this fails ~75% of the time for reasons that
    say nothing about the matcher — which is exactly why the benchmark quotients pose error
    by the symmetry group instead of switching angular scoring off.
    """
    model, pts, nrm = _build(_box())
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=4).as_matrix()
    T[:3, 3] = [0.01, -0.02, 0.03]

    best = match(model, *_apply(T, pts, nrm)).best
    assert best is not None
    pos, ang = _orbit_error(best.T, T, _box_c2_group())
    assert pos < 5e-3, f"translation off by {pos * 1e3:.2f} mm even modulo symmetry"
    assert ang < 10.0, f"rotation off by {ang:.2f} deg even modulo symmetry"


def test_recall_degrades_gracefully_with_occlusion():
    """Bin instances are 10-48% visible, so behaviour under partial views is the operating
    regime, not an edge case. Half a part must still be matchable."""
    model, pts, nrm = _build(_bumpy())
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=11).as_matrix()
    s_pts, s_nrm = _apply(T, pts, nrm)

    keep = s_pts[:, 2] > np.median(s_pts[:, 2])          # keep the camera-facing half
    best = match(model, s_pts[keep], s_nrm[keep]).best
    assert best is not None, "no pose from a 50% visible part"
    pos, ang = _pose_error(best.T, T)
    assert pos < 8e-3 and ang < 15.0, f"half-visible: {pos * 1e3:.2f} mm / {ang:.2f} deg"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Weighted voting — the mechanism the ablation depends on
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["ref", "product", "geometric_mean"])
def test_uniform_weights_reproduce_the_unweighted_result(mode):
    """Weighting must be a strict generalisation of plain voting.

    Without this, an ablation arm that "adds weights" would be confounded by whatever else
    the weighted code path does differently, and a measured difference between arms could
    not be attributed to the weights.
    """
    model, pts, nrm = _build(_bumpy())
    plain = match(model, pts, nrm).best
    weighted = match(model.with_weights(np.ones(model.n_points)), pts, nrm,
                     weight_mode=mode).best

    assert plain is not None and weighted is not None
    assert np.allclose(plain.T, weighted.T, atol=1e-12)
    assert plain.votes == pytest.approx(weighted.votes, rel=1e-9)


def test_weights_actually_change_the_vote_tally():
    """The converse guard: non-uniform weights must reach the accumulator.

    A weight argument that is accepted and then ignored would make every weighted arm of the
    ablation silently identical to the baseline, and the study would report "no effect".
    """
    model, pts, nrm = _build(_bumpy())
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, model.n_points)

    plain = match(model, pts, nrm).best
    weighted = match(model.with_weights(w), pts, nrm).best
    assert plain is not None and weighted is not None
    assert weighted.votes != pytest.approx(plain.votes, rel=1e-6)


def test_negative_weights_are_rejected():
    """A negative vote is not a down-weighting, it is a subtraction from an unrelated
    hypothesis's tally — silently corrupting the accumulator rather than the intended cell."""
    model, _, _ = _build(_bumpy())
    with pytest.raises(ValueError):
        model.with_weights(-np.ones(model.n_points))
    with pytest.raises(ValueError):
        model.with_weights(np.ones(model.n_points + 3))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parameter derivation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_derived_config_is_scale_invariant(scale):
    """A 20 mm bolt and a 400 mm bracket must be configured identically in relative terms.

    Everything that scales with the part is asserted as a *ratio*; the sensor-derived
    quantities deliberately do not scale, because sensor noise is an absolute property of
    the camera and not of the part in front of it.
    """
    mesh = _bumpy()
    mesh.scale(scale, center=(0.0, 0.0, 0.0))
    pts, _ = _cloud(mesh)
    cfg = PPFConfig.derive(pts, **FAST)
    base = PPFConfig.derive(_cloud(_bumpy())[0], **FAST)

    assert cfg.tau / cfg.diameter == pytest.approx(base.tau / base.diameter, rel=0.15)
    assert cfg.n_angle == base.n_angle
    assert cfg.n_alpha == base.n_alpha


def test_tau_hits_the_requested_point_budget():
    """tau is bisected on the real downsampled count rather than a surface-area formula.

    The closed form it replaced, ``tau = sqrt(N*s^2 / M)``, understates area by ~4.5x on a
    Poisson-sampled cloud (the median nearest-neighbour distance is 0.4697/sqrt(density),
    not 1/sqrt(density)). Since the model point count goes as tau^-2 and work as its square,
    that 2x error in tau was a ~20x error in runtime.
    """
    pts, nrm = _cloud(_bumpy())
    for target in (150, 400, 900):
        cfg = PPFConfig.derive(pts, model_target_points=target)
        got = len(downsample(pts, nrm, cfg.tau)[0])
        assert 0.6 * target < got < 1.7 * target, f"asked {target}, got {got}"


def test_under_resolved_parts_are_reported_not_silently_accepted():
    """When the smallest feature is finer than the sensor can see, the honest answer is to
    clamp to the sensor floor *and say so* — that is a real answer to "will this part work",
    which nothing else in the pipeline currently provides."""
    pts, _ = _cloud(_bumpy())
    cfg = PPFConfig.derive(pts, min_feature_size=1e-6, sensor=SensorProfile())
    assert "UNDER-RESOLVED" in cfg.provenance["tau"]


def test_bucket_cap_bounds_a_degenerate_part():
    """A cuboid collapses onto six distinct normals, so its feature bins are enormous —
    12 000 entries before capping, which expands to ~1.8e9 votes for a single instance.

    Deduplication does not help here: it runs after expansion, so it fixes the vote *bias*
    but not the *cost*. The cap is what makes flat parts tractable at all.
    """
    model, _, _ = _build(_box())
    _, counts = np.unique(model.keys, return_counts=True)
    assert counts.max() <= model.cfg.max_bucket_entries


def test_repeated_runs_agree():
    """Determinism is what makes an unattended sweep over a large catalogue trustworthy."""
    model, pts, nrm = _build(_bumpy())
    a, b = match(model, pts, nrm).best, match(model, pts, nrm).best
    assert np.array_equal(a.T, b.T)
    assert a.votes == b.votes


def test_unimplemented_backend_fails_loudly():
    """The CuPy path is Phase 4. Falling back silently to NumPy would let a GPU benchmark
    report CPU timings."""
    model, pts, nrm = _build(_bumpy())
    with pytest.raises(NotImplementedError):
        match(model, pts, nrm, backend="cupy")
