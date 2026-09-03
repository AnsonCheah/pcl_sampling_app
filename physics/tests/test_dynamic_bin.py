"""Dynamic bin sizing: pure-function tests for solve_bin_dim() / obb_packing_factor().

Fast -- no MuJoCo model is compiled and no physics is stepped. Boxes are used throughout
because their OBB volume, footprint and stable-pose height are exact, so every expected bin
dimension below is a closed-form value rather than a fitted one.

Design reference: the bin is solved FIRST and is authoritative; layer count and instance
count are read off the final bin. Height holds LAYERS_AT_FULL_FILL *effective* layers at 100%
fill, where an effective layer is `layer_h / packing` (parts tilt and bridge in a random pile).
"""

import numpy as np
import pytest
import trimesh

from physics.mujoco_bin_scene import (
    MAX_BIN_DIM,
    MIN_BIN_DIM,
    MIN_WALL_THICKNESS,
    MIN_PARTS_PER_LAYER,
    LAYERS_AT_FULL_FILL,
    BIN_TOP_MARGIN_FRAC,
    MIN_AUTO_PARTS,
    MAX_AUTO_PARTS,
    PACKING_FACTOR_BASE,
    PACKING_FACTOR_FLOOR,
    MujocoBinScene,
    obb_packing_factor,
    solve_bin_dim,
    hopper_inward_offset,
)

W0, L0, H0, T0 = MAX_BIN_DIM


def _effective_layer_height(part_mesh):
    """h_eff = layer_h / packing, with layer_h the z-extent of the most probable stable pose."""
    poses = MujocoBinScene.get_stable_poses(part_mesh)
    R0 = poses[0][0]
    verts = np.asarray(part_mesh.vertices)
    rot = (R0 @ verts.T).T
    layer_h = float(rot[:, 2].max() - rot[:, 2].min())
    return layer_h / obb_packing_factor(part_mesh)


# -- packing factor ------------------------------------------------------------


def test_obb_packing_factor_matches_formula(make_box_part):
    """Reproduces the taper formula, read from the OBB's OWN side lengths."""
    for extents in [(0.02, 0.02, 0.02), (0.15, 0.006, 0.006), (0.05, 0.04, 0.005)]:
        part, _ = make_box_part(extents)
        ext = np.sort(part.bounding_box_oriented.primitive.extents)[::-1]
        ar = float(ext[0] / max(ext[2], 1e-9))
        expected = max(PACKING_FACTOR_FLOOR, PACKING_FACTOR_BASE / np.sqrt(ar))
        assert obb_packing_factor(part) == pytest.approx(expected, rel=1e-12)


def test_packing_factor_uses_obb_side_lengths_not_aabb_of_rotated_box(make_box_part):
    """Regression: `bounding_box_oriented.extents` is the AXIS-ALIGNED bounds of the rotated
    OBB box mesh, not the OBB's side lengths (trimesh #871, #1865). Reading it inflates every
    extent toward the box diagonal, collapsing the measured aspect ratio -- so elongated parts
    looked blockier than they are and got too high a packing factor, over-counting them.

    A rotated rod is the sharp case: true AR 25, but the wrong property reads far lower.
    """
    part, _ = make_box_part((0.15, 0.006, 0.006))
    part = part.copy()
    part.apply_transform(
        trimesh.transformations.random_rotation_matrix(rand=np.array([0.3, 0.6, 0.9])))

    true_ext = np.sort(part.bounding_box_oriented.primitive.extents)[::-1]
    wrong_ext = np.sort(part.bounding_box_oriented.extents)[::-1]
    true_ar = true_ext[0] / true_ext[2]
    wrong_ar = wrong_ext[0] / wrong_ext[2]

    assert true_ar == pytest.approx(25.0, rel=0.05)
    assert wrong_ar < true_ar / 2, "expected the wrong property to understate the aspect ratio"
    # The helper must track the true ratio, i.e. sit at the floor for a rod this elongated.
    assert obb_packing_factor(part) == pytest.approx(PACKING_FACTOR_FLOOR, rel=1e-9)


def test_packing_factor_is_rotation_invariant(make_box_part):
    """A part is the same part however it happens to be oriented in its file."""
    part, _ = make_box_part((0.15, 0.006, 0.006))
    rotated = part.copy()
    rotated.apply_transform(
        trimesh.transformations.random_rotation_matrix(rand=np.array([0.1, 0.5, 0.7])))
    assert obb_packing_factor(rotated) == pytest.approx(obb_packing_factor(part), rel=1e-6)


# -- bounds and clipping -------------------------------------------------------


def test_large_part_keeps_max_bin(make_box_part):
    """A part that clips on every axis returns MAX_BIN_DIM -- today's behaviour, unchanged.

    A 120 mm cube exceeds both the height requirement (h_eff*X/(1-margin) > H0) and the
    footprint requirement (A_parts > W0*L0), so every clamp binds.
    """
    part, _ = make_box_part((0.12, 0.12, 0.12))
    bin_dim, n, layers = solve_bin_dim(part, fill_rate=0.2)
    assert bin_dim == pytest.approx(MAX_BIN_DIM, abs=1e-6)
    assert MIN_AUTO_PARTS <= n <= MAX_AUTO_PARTS


def test_small_part_shrinks_bin(cube_part):
    """The 20 mm cube worked example: a much smaller bin and ~10x fewer parts than today."""
    part, _ = cube_part
    bin_dim, n, layers = solve_bin_dim(part, fill_rate=0.2)
    w, l, h, t = bin_dim

    assert (w, l, h) == pytest.approx((0.1448, 0.1114, 0.2419), abs=1e-3)
    assert t == pytest.approx(MIN_WALL_THICKNESS, abs=1e-9)
    assert n == 48
    assert layers == pytest.approx(1.2, abs=0.05)

    # The point of the exercise: far less bin, far fewer parts.
    assert w * l < 0.1 * (W0 * L0)


def test_bin_floor_takes_priority(make_box_part):
    """MIN_BIN_DIM is a hard floor; the parts-per-layer target yields to it, and the
    instance count is then whatever that floored bin produces."""
    part, _ = make_box_part((0.002, 0.002, 0.002))     # 2 mm cube -- floors on every axis
    bin_dim, n, layers = solve_bin_dim(part, fill_rate=0.2)
    w, l, h, t = bin_dim

    assert w >= MIN_BIN_DIM[0] - 1e-9
    assert l == pytest.approx(MIN_BIN_DIM[1], abs=1e-6)   # short side binds first
    assert h == pytest.approx(MIN_BIN_DIM[2], abs=1e-9)
    assert t == pytest.approx(MIN_WALL_THICKNESS, abs=1e-9)
    # Count follows the bin, not the other way round.
    assert n == MAX_AUTO_PARTS


def test_never_exceeds_or_falls_below_bounds(make_box_part):
    """Sweep three decades of part size: the solved bin always lies within [MIN, MAX]."""
    for side in [0.001, 0.005, 0.02, 0.05, 0.12, 0.3]:
        part, _ = make_box_part((side, side, side))
        w, l, h, t = solve_bin_dim(part, fill_rate=0.5)[0]
        assert MIN_BIN_DIM[0] - 1e-9 <= w <= W0 + 1e-9
        assert MIN_BIN_DIM[1] - 1e-9 <= l <= L0 + 1e-9
        assert MIN_BIN_DIM[2] - 1e-9 <= h <= H0 + 1e-9
        assert MIN_WALL_THICKNESS - 1e-9 <= t <= T0 + 1e-9


def test_aspect_ratio_preserved(make_box_part):
    """Footprint aspect is held at exactly W0:L0 wherever no floor/ceiling binds."""
    for side in [0.015, 0.02, 0.03, 0.05]:
        part, _ = make_box_part((side, side, side))
        w, l, _, _ = solve_bin_dim(part, fill_rate=0.2)[0]
        assert w / l == pytest.approx(W0 / L0, rel=1e-9)


# -- layer model ---------------------------------------------------------------


def test_height_holds_layers_at_full_fill(cube_part):
    """At 100% fill an unclipped bin is exactly LAYERS_AT_FULL_FILL effective layers deep."""
    part, _ = cube_part
    _, _, layers = solve_bin_dim(part, fill_rate=1.0)
    assert layers == pytest.approx(LAYERS_AT_FULL_FILL, rel=1e-6)


def test_stacking_depth_is_fill_times_design_layers(make_box_part):
    """Stacking depth is set by fill rate, not bin size: layers == fill * X when unclipped,
    and equals the max bin's depth once H clips at H0 (never worse than the max bin)."""
    part, _ = make_box_part((0.02, 0.02, 0.02))        # unclipped
    for fill in [0.2, 0.5, 1.0]:
        assert solve_bin_dim(part, fill_rate=fill)[2] == pytest.approx(
            fill * LAYERS_AT_FULL_FILL, rel=1e-6)

    big, _ = make_box_part((0.12, 0.12, 0.12))         # H clips at H0
    h_eff = _effective_layer_height(big)
    max_bin_layers = 0.2 * H0 * (1 - BIN_TOP_MARGIN_FRAC) / h_eff
    assert solve_bin_dim(big, fill_rate=0.2)[2] == pytest.approx(max_bin_layers, rel=1e-6)


def test_effective_layer_height_uses_packing(make_box_part):
    """Two parts with identical stable-pose thickness but different packing must NOT get the
    same bin height: the flat one tilts and bridges, so it needs more room per nominal layer.
    Locks the h_eff = layer_h / packing decision."""
    cube, _ = make_box_part((0.005, 0.005, 0.005))       # AR 1   -> packing 0.62
    plate, _ = make_box_part((0.05, 0.04, 0.005))        # AR 10  -> packing ~0.196
    assert obb_packing_factor(plate) < obb_packing_factor(cube)

    h_cube = solve_bin_dim(cube, fill_rate=0.2)[0][2]
    h_plate = solve_bin_dim(plate, fill_rate=0.2)[0][2]
    assert h_plate > h_cube * 2.0     # same layer_h (5 mm), materially taller bin


def test_fill_rate_semantics_unchanged(make_box_part):
    """Feeding the solved bin back through the ORIGINAL _auto_part_count formula must
    reproduce n. Catches an algebra slip in the inversion."""
    for side in [0.008, 0.02, 0.05]:
        part, _ = make_box_part((side, side, side))
        for fill in [0.2, 0.6]:
            (w, l, h, _), n, _ = solve_bin_dim(part, fill_rate=fill)
            packing = obb_packing_factor(part)
            obb_vol = float(part.bounding_box_oriented.volume)
            expected = round(fill * packing * w * l * h * (1 - BIN_TOP_MARGIN_FRAC) / obb_vol)
            expected = int(np.clip(expected, MIN_AUTO_PARTS, MAX_AUTO_PARTS))
            assert n == expected


# -- spawn / pose feasibility --------------------------------------------------


def test_stable_pose_admissible(make_box_part):
    """_compute_valid_stable_poses keeps only poses with lz <= 0.9*bin_height, and
    _sample_constrained_rotation's tilt cone collapses to zero in too shallow a bin.
    Every solved bin must admit at least one pose AND leave a non-zero tilt cone."""
    for extents in [(0.02, 0.02, 0.02), (0.15, 0.006, 0.006), (0.05, 0.04, 0.005),
                    (0.002, 0.002, 0.002), (0.12, 0.12, 0.12)]:
        part, _ = make_box_part(extents)
        h = solve_bin_dim(part, fill_rate=0.2)[0][2]
        verts = np.asarray(part.vertices)

        admissible = []
        for R_stable, _prob in MujocoBinScene.get_stable_poses(part):
            rot = (R_stable @ verts.T).T
            lz = float(rot[:, 2].max() - rot[:, 2].min())
            if lz <= 0.9 * h:
                lx = float(rot[:, 0].max() - rot[:, 0].min())
                ly = float(rot[:, 1].max() - rot[:, 1].min())
                diag = np.hypot(lx, ly)
                theta_max = np.arcsin(np.clip((0.9 * h - lz) / max(diag, 1e-12), 0.0, 1.0))
                admissible.append(theta_max)

        assert admissible, f"no stable pose fits the solved bin for {extents}"
        assert max(admissible) > 0.0, f"tilt cone collapsed for {extents}"


def test_spawn_band_non_empty(make_box_part):
    """_generate_random_scene samples x in [-hx+margin, hx-margin] with
    margin = bounding_sphere_radius + wall thickness. If hx <= margin the band inverts and
    np.random.uniform silently returns poses OUTSIDE the bin."""
    for extents in [(0.002, 0.002, 0.002), (0.02, 0.02, 0.02), (0.15, 0.006, 0.006),
                    (0.05, 0.04, 0.005), (0.12, 0.12, 0.12)]:
        part, _ = make_box_part(extents)
        w, l, _, t = solve_bin_dim(part, fill_rate=0.2)[0]
        margin = part.bounding_sphere.primitive.radius + t
        assert w / 2 > margin, f"x spawn band empty for {extents}"
        assert l / 2 > margin, f"y spawn band empty for {extents}"


def test_hopper_throat_admits_part(make_box_part):
    """The hopper inset is absolute (0.02 m) today; on a shrunken bin that throttles the
    opening below the part footprint and NO part can enter the bin."""
    for extents in [(0.002, 0.002, 0.002), (0.02, 0.02, 0.02), (0.05, 0.04, 0.005)]:
        part, _ = make_box_part(extents)
        bin_dim = solve_bin_dim(part, fill_rate=0.2)[0]
        w, l, _, t = bin_dim
        off = hopper_inward_offset(bin_dim)
        footprint_diag = float(np.hypot(
            *np.sort(part.bounding_box_oriented.primitive.extents)[::-1][:2]))
        assert 2 * (w / 2 - 2 * t - off) > footprint_diag, f"hopper throat too narrow {extents}"
        assert 2 * (l / 2 - 2 * t - off) > footprint_diag, f"hopper throat too narrow {extents}"


def test_hopper_offset_bit_identical_at_max_bin():
    """The bin-relative rewrite must reproduce the old absolute constant at the max bin."""
    assert hopper_inward_offset(MAX_BIN_DIM) == pytest.approx(0.02, abs=1e-12)
