"""Tests for geometry.ambiguity — view-dependent and global ambiguity detection.

Run:  python -m pytest geometry/tests/test_ambiguity.py -q

Two properties matter and both are asserted here.

1. The analysis must reproduce every case the *existing* global symmetry detector already
   gets right (`MM_Optimizer/test_symmetry_detection.py`), or it has been overfitted to
   the view-dependent case that motivated it.  Those parts must additionally come back
   flagged `is_global` with `view_fraction == 1`.

2. It must recover a rotation axis that does **not** pass through the centroid, to a
   tolerance tight enough to build a model frame from.  This is the capability the
   centroid/OBB-based detector structurally cannot have, so it is asserted on the axis
   *point*, not just the direction.

Settings are reduced (fewer viewpoints, lower raycast resolution) so the suite stays
quick; the defaults in `AmbiguityConfig` are what production uses.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import pytest
import trimesh

from geometry.ambiguity import (
    AmbiguityAxis,
    AmbiguityConfig,
    AmbiguityProfile,
    _explains,
    _rotation_about,
    ambiguity_geometries,
    analyse_ambiguity,
    discriminative_colours,
    heat_colour,
    load_ambiguity_profile,
    rank_axes,
    save_ambiguity_profile,
)
from geometry.geom_utils import trimesh_to_o3d

# Reduced from the production defaults purely for test runtime.
FAST = dict(n_views=48, res=144)
N_SAMPLE = 8000


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _centred(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    mesh.compute_vertex_normals()
    return mesh


def _cloud(mesh: o3d.geometry.TriangleMesh, n: int = N_SAMPLE) -> o3d.geometry.PointCloud:
    o3d.utility.random.seed(0)
    return mesh.sample_points_uniformly(n, use_triangle_normal=False)


def _prism(sides: int, radius: float, height: float) -> o3d.geometry.TriangleMesh:
    ring = [(radius * np.cos(a), radius * np.sin(a))
            for a in np.linspace(0.0, 2.0 * np.pi, sides + 1)[:-1]]
    poly = trimesh.path.polygons.Polygon(ring)
    return _centred(trimesh_to_o3d(trimesh.creation.extrude_polygon(poly, height)))


def _analyse(mesh, cfg=None, scale=1.0):
    if scale != 1.0:
        mesh = o3d.geometry.TriangleMesh(mesh)
        mesh.scale(scale, center=(0.0, 0.0, 0.0))
        mesh.compute_vertex_normals()
    return analyse_ambiguity(mesh, _cloud(mesh), cfg or AmbiguityConfig(**FAST))


def _axis_distance(query: np.ndarray, direction: np.ndarray, point: np.ndarray) -> float:
    """Perpendicular distance from ``query`` to the line (direction, point)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    v = np.asarray(query, float) - np.asarray(point, float)
    return float(np.linalg.norm(v - float(v @ d) * d))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Global-symmetry regression — the anti-overfitting guard
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,make,expected_fold", [
    ("cylinder",     lambda: _centred(o3d.geometry.TriangleMesh.create_cylinder(0.020, 0.080, resolution=64)), 0),
    ("sphere",       lambda: _centred(o3d.geometry.TriangleMesh.create_sphere(0.030, resolution=30)), 0),
    ("cone",         lambda: _centred(o3d.geometry.TriangleMesh.create_cone(0.025, 0.060, resolution=60)), 0),
    ("box",          lambda: _centred(o3d.geometry.TriangleMesh.create_box(0.100, 0.030, 0.020)), 2),
    ("tri_prism",    lambda: _prism(3, 0.030, 0.050), 3),
    ("square_prism", lambda: _prism(4, 0.025, 0.050), 4),
    ("hex_prism",    lambda: _prism(6, 0.025, 0.050), 6),
])
def test_global_symmetry_fold_is_recovered(name, make, expected_fold):
    """Fold order must match, and a globally symmetric part must be flagged as global.

    ``fold`` is the value that becomes MechVision's ``angleStep`` (360/fold), so getting
    a hex prism back as C2 would leave two thirds of its ambiguity unmitigated.
    """
    profile = _analyse(make())
    assert profile.dominant is not None, f"{name}: no axis found"
    assert profile.dominant.fold == expected_fold, (
        f"{name}: expected fold {expected_fold}, got {profile.dominant.fold}")
    assert profile.dominant.is_global, f"{name}: global symmetry not flagged as global"
    assert profile.dominant.view_fraction == pytest.approx(1.0, abs=0.05)
    # A globally symmetric part offers nothing to disambiguate with.
    assert profile.discriminative_fraction < 0.05


def test_symmetric_primitive_axes_pass_through_the_centre():
    """Sanity check on the axis point for shapes whose axis is known to be central."""
    for make in (lambda: _centred(o3d.geometry.TriangleMesh.create_cylinder(0.020, 0.080, resolution=64)),
                 lambda: _prism(6, 0.025, 0.050)):
        profile = _analyse(make())
        assert _axis_distance(np.zeros(3), profile.dominant.direction,
                              profile.dominant.point) < 0.0015


def test_sphere_reports_several_independent_axes():
    """A sphere is only partially mitigable — MechVision can use one axis of many."""
    profile = _analyse(_centred(o3d.geometry.TriangleMesh.create_sphere(0.030, resolution=30)))
    assert profile.n_significant_axes >= 3


def test_box_reports_exactly_three_global_axes():
    """A cuboid has three global C2 axes — and legitimately more view-dependent ones.

    Rotating 180 degrees about a face diagonal is not a symmetry of a 30x20 cross
    section, but it does map one visible face patch onto another, so it is a real
    view-dependent ambiguity and is expected to appear.  The assertion is therefore on
    the *global* axes, which are exactly the three box axes.
    """
    profile = _analyse(_centred(o3d.geometry.TriangleMesh.create_box(0.100, 0.030, 0.020)))
    globals_ = [ax for ax in profile.axes if ax.is_global]
    assert len(globals_) == 3
    assert all(ax.fold == 2 for ax in globals_)
    # Each global axis lies along a box axis.
    dirs = np.abs(np.array([ax.direction for ax in globals_]))
    assert np.allclose(dirs.max(axis=1), 1.0, atol=0.02)
    assert np.allclose(np.sort(dirs.argmax(axis=1)), [0, 1, 2])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Off-centroid axis recovery — the capability that motivates the module
# ─────────────────────────────────────────────────────────────────────────────

def _cylinder_with_offset_mass():
    """Cylinder on +Z through the origin, plus a block that drags the centroid away.

    The block breaks global symmetry and moves both the centroid and the AABB centre,
    but the cylinder's own axis stays at x=y=0.  A detector that assumes the axis passes
    through the centroid or the AABB centre cannot get this right.
    """
    cyl = trimesh.creation.cylinder(radius=0.015, height=0.060)
    blk = trimesh.creation.box((0.040, 0.012, 0.012))
    blk.apply_translation((0.033, 0.0, 0.018))
    mesh = trimesh_to_o3d(trimesh.util.concatenate([cyl, blk]))
    mesh.compute_vertex_normals()
    return mesh


def test_recovers_axis_that_misses_the_centroid():
    mesh = _cylinder_with_offset_mass()
    cloud = _cloud(mesh)
    profile = analyse_ambiguity(mesh, cloud, AmbiguityConfig(**FAST))

    true_dir = np.array([0.0, 0.0, 1.0])
    true_pt = np.zeros(3)
    pts = np.asarray(cloud.points)
    centroid = pts.mean(axis=0)
    aabb_centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))

    # The fixture is only meaningful if the axis really does miss both centres.
    assert _axis_distance(centroid, true_dir, true_pt) > 0.005
    assert _axis_distance(aabb_centre, true_dir, true_pt) > 0.010

    match = [ax for ax in profile.axes
             if abs(float(ax.direction @ true_dir)) > np.cos(np.deg2rad(3.0))]
    assert match, "the off-centroid axis direction was not recovered"
    best = max(match, key=lambda ax: ax.view_fraction)
    assert _axis_distance(best.point, true_dir, true_pt) < 0.0015, (
        "axis direction found but its position is wrong — a frame built from this "
        "would rotate about the wrong line")
    assert best.fold == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Negative control and the noise floor
# ─────────────────────────────────────────────────────────────────────────────

def test_random_rotations_stay_below_the_survival_threshold():
    """The separation the f_tau default relies on must actually hold.

    `f_tau` admits a transform explaining >=60% of a visible patch.  That is only sound
    if arbitrary rotations land well below it; this pins the margin so a future tolerance
    change cannot quietly erase it.
    """
    from scipy.spatial import cKDTree
    from geometry.ambiguity import _visibility_masks

    mesh = _centred(o3d.geometry.TriangleMesh.create_box(0.100, 0.030, 0.020))
    cloud = _cloud(mesh)
    pts = np.asarray(cloud.points)
    nrm = np.asarray(cloud.normals)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)

    cfg = AmbiguityConfig(**FAST)
    tree = cKDTree(pts)
    eps = max(cfg.epsilon_spacing_factor * float(np.median(tree.query(pts, k=2)[0][:, 1])),
              cfg.epsilon_floor_m)
    visible, _ = _visibility_masks(mesh, pts, cfg)
    diameter = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    rng = np.random.default_rng(1)
    best = []
    for _ in range(60):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        # Avoid accidentally sampling one of the box's own C2 axes.
        if np.abs(d).max() > 0.98:
            continue
        p = pts.mean(axis=0) + rng.normal(scale=0.25 * diameter, size=3)
        p = p - float(p @ d) * d
        R, t = _rotation_about(d, p, float(rng.uniform(30.0, 330.0)))
        ex = _explains(pts, nrm, tree, nrm, R, t, eps, cfg.normal_cos_tol)
        best.append(max((ex[v].mean() if v.any() else 0.0) for v in visible))

    assert np.percentile(best, 95) < 1.0 - cfg.f_tau, (
        "random rotations reach the survival threshold; f_tau has no margin")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Determinism and scale invariance
# ─────────────────────────────────────────────────────────────────────────────

def test_repeated_runs_agree():
    """Determinism is what makes this safe to run unattended over a large catalogue."""
    mesh = _prism(6, 0.025, 0.050)
    cloud = _cloud(mesh)
    a = analyse_ambiguity(mesh, cloud, AmbiguityConfig(**FAST))
    b = analyse_ambiguity(mesh, cloud, AmbiguityConfig(**FAST))
    assert a.dominant.fold == b.dominant.fold
    assert np.allclose(a.dominant.direction, b.dominant.direction, atol=1e-9)
    assert np.allclose(a.dominant.point, b.dominant.point, atol=1e-9)
    assert a.discriminative_fraction == pytest.approx(b.discriminative_fraction, abs=1e-9)


@pytest.mark.parametrize("offset_m", [0.0, 0.85])
def test_result_is_translation_invariant(offset_m):
    """A part in assembly coordinates must analyse the same as one at the origin.

    This was not true and it silently corrupted the model frame. `25333MB000`'s STL sits
    0.85 m out; analysed there it produced 5 axes with the disc axis split into two
    fragments (area 0.415 + 0.461) that individually lost to a C2 axis, where the same
    cloud centred gives 4 axes and the disc axis dominant at 0.496. The frame was then
    built around the wrong axis.

    Every app path centres the mesh first, so only `bench/generate_scenes.py` hit it —
    which is exactly why relying on callers is not good enough.
    """
    mesh = _cylinder_with_offset_mass()
    cloud = _cloud(mesh)
    if offset_m:
        shift = np.array([-0.6547, 0.3403, 0.4134]) / 0.8458 * offset_m
        mesh = o3d.geometry.TriangleMesh(mesh)
        mesh.translate(shift)
        cloud = o3d.geometry.PointCloud(cloud)
        cloud.translate(shift)

    profile = analyse_ambiguity(mesh, cloud, AmbiguityConfig(**FAST))
    assert profile.dominant is not None

    true_dir = np.array([0.0, 0.0, 1.0])
    match = [ax for ax in profile.axes
             if abs(float(ax.direction @ true_dir)) > np.cos(np.deg2rad(3.0))]
    assert len(match) == 1, (
        f"expected the axis to form ONE group, got {len(match)} — a split here is how the "
        f"wrong axis wins the ranking")

    # The axis point must come back in the CALLER's frame, not the internal centred one.
    true_pt = np.zeros(3) + (shift if offset_m else 0.0)
    assert _axis_distance(match[0].point, true_dir, true_pt) < 0.0015


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_result_is_scale_invariant(scale):
    """A 20 mm and a 500 mm part must be judged the same way.

    This is what the relative tolerances buy; an absolute area threshold would make the
    verdict depend on part size.
    """
    base = _prism(6, 0.025, 0.050)
    profile = _analyse(base, scale=scale)
    assert profile.dominant is not None
    assert profile.dominant.fold == 6
    assert profile.dominant.view_fraction == pytest.approx(1.0, abs=0.05)
    # The tolerance tracks the part until the sensor-physics floor takes over — which is
    # the intended behaviour, not a scale-invariance failure: below the floor the sensor
    # cannot resolve the agreement being asked about.
    cfg = AmbiguityConfig(**FAST)
    expected = max(_analyse(base).epsilon_m * scale, cfg.epsilon_floor_m)
    assert profile.epsilon_m == pytest.approx(expected, rel=0.15)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Persistence round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_profile_round_trips_through_json(tmp_path):
    profile = _analyse(_prism(6, 0.025, 0.050))
    path = tmp_path / "ambiguity_profile.json"
    save_ambiguity_profile(profile, path)
    back = load_ambiguity_profile(path)

    assert isinstance(back, AmbiguityProfile)
    assert back.dominant is not None
    assert back.dominant.fold == profile.dominant.fold
    assert np.allclose(back.dominant.direction, profile.dominant.direction)
    assert np.allclose(back.dominant.point, profile.dominant.point)
    assert back.dominant.angle_step_deg() == pytest.approx(60.0)
    assert len(back.per_view) == len(profile.per_view)


def test_angle_step_matches_the_fold():
    profile = _analyse(_prism(4, 0.025, 0.050))
    assert profile.dominant.fold == 4
    assert profile.dominant.angle_step_deg() == pytest.approx(90.0)


def _write_profile(path, axes, exponent=None):
    """Hand-write a sidecar so the stored order can be made to disagree with the metrics."""
    payload = {
        "axes": axes,
        "dominant": axes[0],
        "n_significant_axes": len(axes),
        "discriminative_fraction": 0.5,
        "frame_changed": False,
        "ppf_degeneracy": {},
        "epsilon_m": 0.0025,
        "f_tau": 0.40,
        "diameter_m": 0.12,
        "per_view": [],
    }
    if exponent is not None:                 # absent key vs present-but-null are both real
        payload["rank_area_exponent"] = exponent
    path.write_text(json.dumps(payload, indent=2))
    return path


def _axis_record(fold, views, area, direction):
    return {"direction": list(direction), "point": [0.0, 0.0, 0.0], "fold": fold,
            "angles_deg": [180.0] if fold == 2 else [], "is_global": False,
            "view_fraction": views, "area_fraction": area,
            "patch_fraction": 0.65, "best_patch_fraction": 0.70,
            "score": views}          # the OLD scheme: score == view_fraction


# The measured 25333MB000 case: a C2 axis with more views but less coverage, against the
# off-centroid disc axis that actually flips in real scenes.
_C2   = _axis_record(2, 0.14, 0.36, (0.0, 0.0, 1.0))
_DISC = _axis_record(1, 0.10, 0.50, (1.0, 0.0, 0.0))


@pytest.mark.parametrize("exponent", [None, 2.0])
def test_loader_reranks_a_stale_sidecar(tmp_path, exponent):
    """A sidecar's stored order must not survive into `dominant`.

    Written before the ranking exponent existed, `25333MB000`'s sidecar was ordered by the
    old `score = view_fraction` and named the C2 axis dominant. `dominant` is what selects
    MechVision's rotationStrategy and angleStep, so trusting the stored order ships the
    wrong axis — re-ranking the same stored numbers is what puts the disc axis first.
    """
    path = _write_profile(tmp_path / "ambiguity_profile.json", [_C2, _DISC], exponent)
    back = load_ambiguity_profile(path)

    assert back.rank_area_exponent == 2.0        # a null/absent key must reach the default
    # views*area^2: disc 0.10*0.25 = 0.0250 beats C2 0.14*0.1296 = 0.0181
    assert back.dominant is not None
    assert np.allclose(np.abs(back.dominant.direction), [1.0, 0.0, 0.0]), \
        "loaded profile kept the stale write-time order instead of re-ranking"
    assert back.axes[0].area_fraction == pytest.approx(0.50)


def test_loader_honours_a_deliberate_exponent(tmp_path):
    """A file that recorded its own exponent keeps that intent; only a file that never
    recorded one adopts the current default."""
    path = _write_profile(tmp_path / "ambiguity_profile.json", [_C2, _DISC], exponent=0.0)
    back = load_ambiguity_profile(path)

    assert back.rank_area_exponent == 0.0        # 0.0 is meaningful, not a missing value
    # At k=0 the score is view_fraction alone, so the C2 axis leads again.
    assert np.allclose(np.abs(back.dominant.direction), [0.0, 0.0, 1.0])


def test_rank_axes_is_reversible_on_a_loaded_profile(tmp_path):
    """Re-ranking needs no re-analysis — that is what makes the exponent cheap to tune."""
    path = _write_profile(tmp_path / "ambiguity_profile.json", [_C2, _DISC])
    back = load_ambiguity_profile(path)
    assert np.allclose(np.abs(back.dominant.direction), [1.0, 0.0, 0.0])

    rank_axes(back, 0.0)
    assert np.allclose(np.abs(back.dominant.direction), [0.0, 0.0, 1.0])
    rank_axes(back, 2.0)
    assert np.allclose(np.abs(back.dominant.direction), [1.0, 0.0, 0.0])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_heat_colour_is_bounded_and_ordered():
    ramp = heat_colour(np.linspace(0.0, 1.0, 32))
    assert ramp.shape == (32, 3)
    assert ramp.min() >= 0.0 and ramp.max() <= 1.0
    # Brightness must increase with the score, or "hot = discriminative" reads backwards.
    assert np.all(np.diff(ramp.sum(axis=1)) > -1e-9)
    assert np.allclose(heat_colour(0.5), heat_colour([0.5]).reshape(3))


def test_discriminative_colours_stay_visible_at_the_cool_end():
    """The most ambiguous points must not render as near-black on a dark background."""
    profile = _analyse(_centred(o3d.geometry.TriangleMesh.create_cylinder(
        0.020, 0.080, resolution=64)))
    colours = discriminative_colours(profile)
    assert len(colours) == len(profile.per_point_discriminative)
    assert colours.min() >= 0.0 and colours.max() <= 1.0
    assert colours.sum(axis=1).min() > 0.10


def test_ambiguity_geometries_reports_the_offset_it_draws():
    """The legend's offsets must describe the axes actually rendered."""
    mesh = _cylinder_with_offset_mass()
    cloud = _cloud(mesh)
    profile = analyse_ambiguity(mesh, cloud, AmbiguityConfig(**FAST))
    geoms, legend = ambiguity_geometries(cloud, profile, max_axes=3)

    assert legend, "expected at least one axis in the legend"
    assert len(legend) <= 3
    # cloud + (rod + knob) per axis + centroid and AABB markers
    assert len(geoms) == 1 + 2 * len(legend) + 2
    assert geoms[0].has_colors()

    centroid = np.asarray(cloud.points).mean(axis=0)
    for row in legend:
        expected = _axis_distance(centroid, row["direction"], row["point"]) * 1000.0
        assert row["offset_from_centroid_mm"] == pytest.approx(expected, abs=1e-6)
    # Rank 0 is the hottest colour.
    assert legend[0]["colour"].sum() >= legend[-1]["colour"].sum()


def test_ambiguity_geometries_handles_a_part_with_no_axes():
    empty = AmbiguityProfile(per_point_discriminative=np.ones(64))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.random.default_rng(0).normal(size=(64, 3)))
    geoms, legend = ambiguity_geometries(cloud, empty)
    assert legend == []
    assert len(geoms) == 3          # cloud plus the two reference markers


# ---------------------------------------------------------------------------
# Ranking: score decides, fold only breaks ties between global axes
#
# These are unit tests on `rank_axes` with hand-built axes rather than on a whole
# analysis, because the failure they pin is a knife edge: on 25333MB000 the leading
# view-dependent axis beat the runner-up by 1% of score, and an absolute tie window
# swallowed the difference. Driving it from a mesh would make the margin an accident of
# sampling instead of the thing under test.
# ---------------------------------------------------------------------------

def _axis(view_fraction, area_fraction, fold, is_global):
    return AmbiguityAxis(direction=np.array([0.0, 0.0, 1.0]),
                         point=np.zeros(3), fold=fold, angles_deg=[],
                         is_global=is_global,
                         view_fraction=view_fraction, area_fraction=area_fraction)


def test_view_dependent_axes_rank_by_score_not_fold():
    """The measured 25333MB000 case: six view-dependent axes scoring 0.0017-0.0172, the
    best of them fold 1. It must lead. Under an absolute `round(score, 2)` tie window all
    six collapsed into one bucket and the two lower-scoring C2 axes were promoted."""
    profile = AmbiguityProfile(axes=[
        _axis(0.150, 0.336896, 2, False),      # score 0.017025
        _axis(0.140, 0.334580, 2, False),      # score 0.015672
        _axis(0.075, 0.478760, 1, False),      # score 0.017191  <- highest
        _axis(0.105, 0.312494, 2, False),      # score 0.010254
        _axis(0.025, 0.257458, 2, False),
        _axis(0.055, 0.269926, 2, False),
    ])
    rank_axes(profile, area_exponent=2.0)
    assert profile.dominant is profile.axes[0]
    assert profile.axes[0].fold == 1
    assert profile.axes[0].score == pytest.approx(0.017191, abs=1e-6)
    scores = [ax.score for ax in profile.axes]
    assert scores == sorted(scores, reverse=True)


def test_global_axes_still_prefer_the_higher_fold_on_a_tie():
    """A hex prism's C2 and C6 axes are both global and score within a thousandth of each
    other; reporting C2/180deg would leave two thirds of the ambiguity unmitigated."""
    profile = AmbiguityProfile(axes=[
        _axis(1.0, 0.9992, 2, True),
        _axis(1.0, 0.9989, 6, True),
    ])
    rank_axes(profile, area_exponent=2.0)
    assert profile.dominant.fold == 6


def test_a_continuous_global_axis_outranks_a_tied_finite_one():
    profile = AmbiguityProfile(axes=[
        _axis(1.0, 0.999, 2, True),
        _axis(1.0, 0.998, 0, True),
    ])
    rank_axes(profile, area_exponent=2.0)
    assert profile.dominant.fold == 0


def test_fold_preference_never_overrides_a_real_score_gap():
    """Outside the tie band the higher fold must not win, global or not."""
    profile = AmbiguityProfile(axes=[
        _axis(1.00, 1.00, 1, True),            # score 1.00
        _axis(0.50, 1.00, 6, True),            # score 0.50 -- far outside the band
    ])
    rank_axes(profile, area_exponent=2.0)
    assert profile.dominant.fold == 1


def test_ranking_is_unchanged_by_a_uniform_rescale_of_scores():
    """The tie window is a fraction of the leading score, so the same axes ordered the same
    way must come out the same whether they score near 1.0 or near 0.01."""
    big = AmbiguityProfile(axes=[_axis(1.0, 0.999, 2, True), _axis(1.0, 0.998, 6, True)])
    rank_axes(big, area_exponent=2.0)
    small = AmbiguityProfile(axes=[_axis(0.01, 0.999, 2, True), _axis(0.01, 0.998, 6, True)])
    rank_axes(small, area_exponent=2.0)
    assert big.dominant.fold == small.dominant.fold == 6


def test_profile_records_the_exponent_it_ranked_with():
    mesh = _centred(o3d.geometry.TriangleMesh.create_box(0.100, 0.030, 0.020))
    cfg = AmbiguityConfig(n_views=8, res=64, rank_area_exponent=0.0)
    profile = analyse_ambiguity(mesh, _cloud(mesh, 1500), cfg)
    assert profile.rank_area_exponent == 0.0
