"""Tests for registration.ppf — the standalone vanilla PPF + Hough matcher.

Run:  python -m pytest registration/tests/test_ppf.py -q

The properties asserted here are the ones that, if they silently broke, would invalidate any
result the matcher produces:

1. **The package is standalone.**  Its whole reason for existing is that it can be lifted
   out of this repository unchanged, so an accidental ``from geometry import ...`` is a
   correctness bug, not a style one — and it would not otherwise be caught, because the repo
   is always on ``sys.path`` when the suite runs.
2. **The frame convention round-trips.**  PPF encodes a pose as (model point, alpha); train,
   match and pose reconstruction must share one convention.  A sign error there still
   produces plausible-looking poses, just wrong ones.
3. **Symmetric parts come back in the right symmetry orbit**, not at one nominated pose.
   Scoring a box against a single ground-truth rotation would fail three quarters of the
   time for reasons that have nothing to do with the matcher.
4. **Derived parameters are scale-invariant.**  A 20 mm part and a 400 mm part must be
   treated the same way, or the "no per-part configuration" claim is empty.
5. **The weighting API is gone**, not merely unused.  Weighted voting lives in
   ``registration.ppf_saliency``; if a weights argument silently survived here, the two
   packages would stop being the distinct things they are meant to be.

Point counts are kept small so the suite stays quick; production uses larger clouds.
"""
import ast
import glob
import inspect
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
    cupy_available,
    downsample,
    frames_to_x,
    match,
    match_many,
    pose_from_correspondence,
)

FAST = dict(model_target_points=220)
N_SAMPLE = 6000

_REGISTRATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The copyable unit: the vanilla matcher plus the code it shares with ppf_saliency. Copying
# it into another project means copying BOTH directories. It is deliberately not all of
# ``registration/`` — ``ppf_saliency`` imports ``geometry`` and ``registration.ppf.bench``
# absolutely, by design, and so cannot travel alone.
_PKG_DIRS = [os.path.join(_REGISTRATION_DIR, "ppf"),
             os.path.join(_REGISTRATION_DIR, "_shared")]

# Everything in this repository the unit must not reach for. "registration" is on the list
# too: an absolute ``registration.ppf.bench.dataset`` import inside it would work here and
# nowhere else, which is the coupling the extraction removed. ``_shared`` is reached by
# relative import (``from .._shared import ...``), which this walk allows.
_REPO_PACKAGES = {"geometry", "sensor", "physics", "stages", "bench", "enums", "app",
                  "registration", "MM_Optimizer"}


def _package_sources():
    """Every .py in the copyable unit, including ppf's bench subpackage."""
    return sorted(f for d in _PKG_DIRS
                  for f in glob.glob(os.path.join(d, "**", "*.py"), recursive=True))


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
# 0. The package is standalone
# ─────────────────────────────────────────────────────────────────────────────

def test_package_imports_nothing_from_this_repository():
    """The extraction claim, asserted statically over every import in the copyable unit.

    A runtime import check cannot catch this: the repo root is on ``sys.path`` for the whole
    test session, so ``from geometry import ...`` would simply succeed and the package would
    look standalone right up until someone copied it elsewhere. The AST walk sees the imports
    whether or not they are reachable, including the ones tucked inside functions.
    """
    offenders = []
    for path in _package_sources():
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import — within the package, which is fine.
                names = [] if node.level else [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in _REPO_PACKAGES:
                    offenders.append(f"{os.path.basename(path)}:{node.lineno} imports {name}")
    assert not offenders, "registration.ppf must not import from the repo:\n" + "\n".join(offenders)


def test_third_party_dependencies_are_only_numpy_scipy_open3d():
    """Pinning the dependency surface, so 'standalone' keeps meaning something.

    Adding, say, scikit-learn for one convenience call would still pass the repo-import test
    above while quietly making the package much harder to lift into another project.
    """
    allowed = {"numpy", "scipy", "open3d",
               # stdlib the harness and the thread pool need; none constrain portability
               "time", "os", "re", "json", "argparse", "concurrent",
               "dataclasses", "typing", "__future__",
               # cupy is OPTIONAL and guarded -- see the separate test below
               "cupy"}
    found = set()
    for path in _package_sources():
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                found.add((node.module or "").split(".")[0])
    assert found <= allowed, f"unexpected dependencies: {sorted(found - allowed)}"


def test_cupy_is_an_optional_dependency_not_a_required_one():
    """CuPy must never be imported at module scope anywhere in the copyable unit.

    A top-level ``import cupy`` would make the whole package fail to import on any machine
    without a CUDA build — turning an optional accelerator into a hard requirement, which is
    exactly the portability the standalone split exists to protect. It is therefore confined
    to one guarded call inside ``_backend.py``.
    """
    offenders = []
    for path in _package_sources():
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [(node.module or "").split(".")[0]]
            if "cupy" not in names:
                continue
            # col_offset == 0 means module scope; anything indented is inside a function.
            if node.col_offset == 0:
                offenders.append(f"{os.path.basename(path)}:{node.lineno} top-level cupy import")
    assert not offenders, "cupy must stay optional:\n" + "\n".join(offenders)


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

    5 mm / 10 deg is the repo's LOOSE gate. Coarse PPF is not expected to reach the
    2 mm / 5 deg TIGHT gate unaided — that is what a fine refinement stage is for — so
    asserting the tight gate here would encode a false expectation of the algorithm.
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
    say nothing about the matcher — which is exactly why a benchmark must quotient pose error
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


def test_noisy_scene_is_still_matched():
    """Depth noise is the operating condition, not an edge case.

    Perturbing at the sensor's own 1-sigma checks that the derived angular binning really is
    wide enough to keep a correct correspondence inside its own bin — the thing
    ``PPFConfig.derive`` claims when it converts depth noise into a bin width.
    """
    model, pts, nrm = _build(_bumpy())
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=5).as_matrix()
    T[:3, 3] = [0.02, 0.01, -0.03]
    s_pts, s_nrm = _apply(T, pts, nrm)

    rng = np.random.default_rng(0)
    s_pts = s_pts + rng.normal(0.0, SensorProfile().sigma_z(), s_pts.shape)

    best = match(model, s_pts, s_nrm).best
    assert best is not None, "no pose under 1-sigma sensor noise"
    pos, ang = _pose_error(best.T, T)
    assert pos < 5e-3 and ang < 10.0, f"noisy: {pos * 1e3:.2f} mm / {ang:.2f} deg"


# ─────────────────────────────────────────────────────────────────────────────
# 3. The weighting API is gone, not merely unused
# ─────────────────────────────────────────────────────────────────────────────

def test_vote_weighting_api_is_absent():
    """Vanilla means the weighting machinery is not reachable from here.

    Left in place but undocumented, it would be used by accident and the split between this
    package and ``registration.ppf_saliency`` would stop meaning anything. The ablation that
    justified the split (2789 T-LESS instances; uniform 0.66 BOP recall vs 0.56 for the
    ambiguity heat map) lives with the code it measured.
    """
    import registration.ppf as pkg

    model, _, _ = _build(_bumpy())
    for gone in ("with_weights", "weights", "entry_point2"):
        assert not hasattr(model, gone), f"PPFModel still carries {gone!r}"
    for gone in ("saliency", "ppf_saliency", "transfer_weights", "combine"):
        assert not hasattr(pkg, gone), f"package still exports {gone!r}"
    assert "weight_mode" not in inspect.signature(match).parameters
    assert not os.path.exists(os.path.join(_REGISTRATION_DIR, "ppf", "saliency.py"))


def test_the_saliency_package_still_exists_and_is_separate():
    """The extension was moved, not deleted — it is the only way to re-test the weighting
    question on a new part catalogue, which is why it was kept at all."""
    from registration.ppf_saliency import PPFModel as WeightedModel
    from registration.ppf_saliency.saliency import ppf_saliency

    assert hasattr(WeightedModel, "with_weights")
    assert callable(ppf_saliency)


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


def test_provenance_records_which_bound_bound():
    """"Why is tau 4.6 mm" must have an answer without re-deriving it by hand — that is the
    whole point of deriving parameters instead of tuning them."""
    pts, _ = _cloud(_bumpy())
    cfg = PPFConfig.derive(pts, **FAST)
    assert cfg.provenance["tau"]
    assert cfg.provenance["angle_bin"]
    assert "tau" in cfg.describe()


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


def test_key_index_agrees_with_binary_search():
    """The direct index is a cache of what ``searchsorted`` would return, so it must agree
    exactly — including on keys whose bin is *empty*, where ``lo == hi``.

    Empty bins are the interesting case and the easy one to get wrong: they are the majority
    of the key space, they never appear in ``model.keys``, and a scene pair landing in one
    must produce a zero-length range rather than an arbitrary neighbour's entries.
    """
    from registration.ppf.model import _build_key_offsets

    model, _, _ = _build(_bumpy())
    assert model.key_offsets is not None, "index should have been built for this part"

    na = model.cfg.n_angle + 1
    keyspace = model.n_dist_bins * na ** 3
    rng = np.random.default_rng(0)
    probe = np.concatenate([
        model.keys[rng.integers(0, len(model.keys), 500)],   # populated bins
        rng.integers(0, keyspace, 500),                      # mostly empty bins
        np.array([0, keyspace - 1]),                         # the boundaries
    ])
    lo, hi = model.key_offsets[probe], model.key_offsets[probe + 1]
    assert np.array_equal(lo, np.searchsorted(model.keys, probe, side="left"))
    assert np.array_equal(hi, np.searchsorted(model.keys, probe, side="right"))

    # And the fallback path must return the same thing, so a part that trips the size cap
    # behaves identically rather than merely not crashing.
    assert _build_key_offsets(model.keys, model.n_dist_bins, 400) is None
    unindexed = PPFModel(**{**model.__dict__, "key_offsets": None})
    a, b = model.lookup(probe)
    c, d = unindexed.lookup(probe)
    assert np.array_equal(a, c) and np.array_equal(b, d)


def test_matching_is_unaffected_by_whether_the_index_was_built():
    """The index is an optimisation, so it must not be observable in the result.

    Asserted on the pose rather than on the ranges, because that is the property anyone
    actually depends on — a lookup bug that shifted entries by one would still produce
    plausible ranges and silently wrong poses.
    """
    model, pts, nrm = _build(_bumpy())
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=9).as_matrix()
    T[:3, 3] = [0.02, -0.01, 0.04]
    s_pts, s_nrm = _apply(T, pts, nrm)

    with_idx = match(model, s_pts, s_nrm, top_k=3)
    without = match(PPFModel(**{**model.__dict__, "key_offsets": None}), s_pts, s_nrm, top_k=3)

    assert len(with_idx.poses) == len(without.poses)
    for a, b in zip(with_idx.poses, without.poses):
        assert np.array_equal(a.T, b.T)
        assert a.votes == b.votes and a.score == b.score
    assert with_idx.n_votes == without.n_votes


def test_repeated_runs_agree():
    """Determinism is what makes an unattended sweep over a large catalogue trustworthy."""
    model, pts, nrm = _build(_bumpy())
    a, b = match(model, pts, nrm).best, match(model, pts, nrm).best
    assert np.array_equal(a.T, b.T)
    assert a.votes == b.votes


def test_unknown_backend_is_rejected_but_a_missing_gpu_is_not():
    """A typo must raise; an absent GPU must not.

    These are different failures and deserve different handling. ``backend="numpu"`` is a
    programming error that silently running on the CPU would hide forever. A machine without
    CUDA is an ordinary deployment fact, and refusing to match there would make the package
    useless on exactly the hardware most people have.

    What keeps the fallback honest is that it is *reported*: ``MatchResult.backend`` says what
    actually ran, so a benchmark cannot pass CPU timings off as GPU ones. That was the real
    concern behind the old hard failure.
    """
    model, pts, nrm = _build(_bumpy())
    with pytest.raises(ValueError):
        match(model, pts, nrm, backend="numpu")

    for requested in ("cupy", "auto"):
        res = match(model, pts, nrm, backend=requested)
        assert res.backend in ("numpy", "cupy")
        if not cupy_available():
            assert res.backend == "numpy", "must fall back when there is no usable GPU"
    assert match(model, pts, nrm, backend="numpy").backend == "numpy"


@pytest.mark.skipif(not cupy_available(), reason="no usable CUDA device")
def test_cupy_backend_agrees_with_numpy():
    """The GPU path must be a pure reimplementation, not a different algorithm.

    Votes and pose counts are asserted exactly — the vote stage is integer bookkeeping and
    must match to the last vote. Pose values get a small tolerance: the GPU reduces in a
    different order, so the accumulator's float sums can differ in the last bits, which can
    move a cluster mean by ~1e-12.
    """
    model, pts, nrm = _build(_bumpy())
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=6).as_matrix()
    T[:3, 3] = [0.03, -0.02, 0.01]
    s_pts, s_nrm = _apply(T, pts, nrm)

    cpu = match(model, s_pts, s_nrm, top_k=3, backend="numpy")
    gpu = match(model, s_pts, s_nrm, top_k=3, backend="cupy")

    assert gpu.backend == "cupy"
    assert gpu.n_scene_points == cpu.n_scene_points
    assert gpu.n_pair_evals == cpu.n_pair_evals
    assert gpu.n_votes == cpu.n_votes, "the GPU cast a different number of votes"
    assert len(gpu.poses) == len(cpu.poses)
    for a, b in zip(cpu.poses, gpu.poses):
        assert np.allclose(a.T, b.T, atol=1e-9), "pose differs beyond float reassociation"
        assert a.votes == pytest.approx(b.votes, rel=1e-12)
        assert a.score == pytest.approx(b.score, rel=1e-12)


def test_match_many_matches_the_serial_result():
    """Parallelism must be an optimisation, not a semantic change: same results, same order.

    Order matters as much as values — results are joined against instance ground truth by
    position, so a pool that returned them out of order would silently mis-score everything.
    """
    from registration.ppf import match_many

    model, pts, nrm = _build(_bumpy())
    clusters = []
    for seed in range(6):
        T = np.eye(4)
        T[:3, :3] = Rot.random(random_state=seed).as_matrix()
        T[:3, 3] = np.random.default_rng(seed).uniform(-0.04, 0.04, 3)
        clusters.append(_apply(T, pts, nrm))

    serial = [match(model, p, n) for p, n in clusters]
    for workers in (1, 4):
        got = match_many(model, clusters, workers=workers)
        assert len(got) == len(serial)
        for a, b in zip(serial, got):
            assert np.array_equal(a.best.T, b.best.T), f"workers={workers} changed the pose"
            assert a.best.votes == b.best.votes
    assert match_many(model, [], workers=4) == []


# ─────────────────────────────────────────────────────────────────────────────
# 5. The benchmark harness that ships with the package
# ─────────────────────────────────────────────────────────────────────────────

def _sym_group_tless_like():
    """A models_info entry shaped like BOP's: one discrete C2 plus a continuous z axis."""
    c2 = np.eye(4)
    c2[:3, :3] = Rot.from_rotvec([0.0, 0.0, np.pi]).as_matrix()
    return {"symmetries_discrete": [c2.reshape(-1).tolist()],
            "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]}


def test_mssd_is_zero_for_a_pose_inside_the_symmetry_orbit():
    """The defining property. If a symmetry rotation cost MSSD anything, every symmetric part
    would be scored as a failure and the benchmark would measure symmetry, not the matcher."""
    from registration.ppf.bench.metrics import mssd, symmetry_transforms_from_bop

    pts, _ = _cloud(_box(), n=800)
    syms = symmetry_transforms_from_bop(
        {"symmetries_discrete": [
            np.block([[Rot.from_rotvec([0, 0, np.pi]).as_matrix(), np.zeros((3, 1))],
                      [np.zeros((1, 3)), np.ones((1, 1))]]).reshape(-1).tolist()]},
        units_to_m=1.0)

    R_gt, t_gt = np.eye(3), np.zeros(3)
    R_est = Rot.from_rotvec([0, 0, np.pi]).as_matrix()      # exactly the symmetry
    assert mssd(R_est, t_gt, R_gt, t_gt, pts, syms) < 1e-9
    # ...and a rotation that is NOT a symmetry must still be charged for.
    R_bad = Rot.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
    assert mssd(R_bad, t_gt, R_gt, t_gt, pts, syms) > 1e-3


def test_metrics_agree_with_bop_toolkit():
    """Our MSSD/ADD/ADI are written here rather than imported, to keep the dependency surface
    at numpy/scipy/open3d. That is only defensible if they agree with the reference.

    Skipped where ``bop_toolkit_lib`` is not installed — it is not a dependency of this
    package, and must never become one.
    """
    bop_pose = pytest.importorskip("bop_toolkit_lib.pose_error")
    bop_misc = pytest.importorskip("bop_toolkit_lib.misc")
    from registration.ppf.bench.metrics import add, adi, mssd, symmetry_transforms_from_bop

    pts, _ = _cloud(_bumpy(), n=500)
    info = _sym_group_tless_like()
    # Compare in one unit system; BOP's own helper does no rescaling, so neither do we here.
    ours = symmetry_transforms_from_bop(info, units_to_m=1.0, max_disc_step=0.5)
    theirs = bop_misc.get_symmetry_transformations(info, max_sym_disc_step=0.5)
    assert len(ours) == len(theirs), f"{len(ours)} symmetry transforms vs BOP's {len(theirs)}"

    rng = np.random.default_rng(0)
    for seed in range(5):
        R_e = Rot.random(random_state=seed).as_matrix()
        t_e = rng.uniform(-0.02, 0.02, 3)
        R_g = Rot.random(random_state=seed + 50).as_matrix()
        t_g = rng.uniform(-0.02, 0.02, 3)
        te, tg = t_e.reshape(3, 1), t_g.reshape(3, 1)

        assert mssd(R_e, t_e, R_g, t_g, pts, ours) == pytest.approx(
            float(bop_pose.mssd(R_e, te, R_g, tg, pts, theirs)), rel=1e-9, abs=1e-12)
        assert add(R_e, t_e, R_g, t_g, pts) == pytest.approx(
            float(bop_pose.add(R_e, te, R_g, tg, pts)), rel=1e-9, abs=1e-12)
        assert adi(R_e, t_e, R_g, t_g, pts) == pytest.approx(
            float(bop_pose.adi(R_e, te, R_g, tg, pts)), rel=1e-9, abs=1e-12)


def test_bop_translation_units_are_converted():
    """models_info offsets are millimetres and everything here is metres. A missed conversion
    is a 1000x error that reads as a gross pose failure rather than as a unit bug."""
    from registration.ppf.bench.metrics import symmetry_transforms_from_bop

    m = np.eye(4)
    m[:3, :3] = Rot.from_rotvec([0.0, 0.0, np.pi]).as_matrix()
    m[:3, 3] = [10.0, 0.0, 0.0]                   # 10 mm
    syms = symmetry_transforms_from_bop({"symmetries_discrete": [m.reshape(-1).tolist()]})
    offsets = [float(np.linalg.norm(s["t"])) for s in syms]
    assert pytest.approx(0.010, rel=1e-9) == max(offsets), "10 mm should become 0.010 m"


def test_evaluate_pose_and_summarise_round_trip():
    """A perfect pose must score zero and pass every gate; a grossly wrong one must fail."""
    from registration.ppf.bench.metrics import TIGHT, evaluate_pose, summarise

    pts, _ = _cloud(_bumpy(), n=400)
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=2).as_matrix()
    T[:3, 3] = [0.01, 0.02, 0.03]

    good = evaluate_pose(T, T, pts, [], diameter=0.1)
    assert good.mssd < 1e-12 and good.te < 1e-12
    assert good.passes(TIGHT) and good.passes_bop()

    T_bad = T.copy()
    T_bad[:3, 3] += 0.05
    bad = evaluate_pose(T_bad, T, pts, [], diameter=0.1)
    assert not bad.passes(TIGHT) and not bad.passes_bop()

    s = summarise([good, bad])
    assert s["n"] == 2 and s["recall_bop"] == pytest.approx(0.5)


def test_benchmark_harness_is_importable_and_tolerates_a_missing_scene_root():
    """The harness ships inside the package, so it has to survive being pointed at nothing —
    a fresh checkout has no scenes until the generator has been run."""
    from registration.ppf.bench import run as bench_run
    from registration.ppf.bench.dataset import list_parts, list_scenes

    assert list_parts(os.path.join(os.path.dirname(__file__), "no_such_root")) == []
    assert list_scenes("obj_000001", os.path.join(os.path.dirname(__file__), "nope")) == []
    assert bench_run.load_models_info(None) == {}
    assert bench_run.symmetry_class(None) == "unknown"
    assert bench_run.symmetry_class({"symmetries_continuous": [1]}) == "continuous"
    assert bench_run.symmetry_class({"symmetries_discrete": [1]}) == "discrete"
    assert bench_run.symmetry_class({}) == "asymmetric"
