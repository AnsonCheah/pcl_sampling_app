"""Dynamic bin sizing: settling behaviour in a shrunken bin.

Slow -- these compile a MuJoCo model and run a full settle. They are the regression tests for
the anti-escape audit: several spawn/drop constants are absolute metres that are invisible on
the 0.76 m max bin and fatal on a 0.15 m one.
"""

import numpy as np
import pytest

from physics.mujoco_bin_scene import (
    MAX_BIN_DIM,
    OVERSHOOT_TOL_LAYERS,
    MujocoBinScene,
    obb_packing_factor,
    solve_bin_dim,
    stable_layer_heights,
)
from physics.tests.test_bin_scene import _check_no_penetration

pytestmark = pytest.mark.slow

# Cap parts for test runtime; the solved count for a 20 mm cube at 20% fill is 48, and the
# behaviours under test (escape, spill) do not need the full pile to manifest.
TEST_PART_CAP = 24


def _effective_layer_height(part_mesh):
    layer_h, _ = stable_layer_heights(part_mesh)
    return layer_h / obb_packing_factor(part_mesh)


def _settle(part, convex, bin_dim, n_parts, settle_time=5.0):
    scene = MujocoBinScene(part, convex, n_parts=n_parts, bin_dim=bin_dim,
                           settle_time=settle_time, render=False)
    scene.simulate()
    return scene


def test_small_part_settles_inside_shrunken_bin(cube_part):
    """No part may escape a bin sized down to ~1/5 of the max footprint."""
    part, convex = cube_part
    bin_dim, n, _ = solve_bin_dim(part, fill_rate=0.2)
    scene = _settle(part, convex, bin_dim, min(n, TEST_PART_CAP))

    report = scene.verify_parts_in_bin()
    assert report["n_out"] == 0, f"{report['n_out']} part(s) escaped: {report['out_of_bin'][:5]}"

    state = scene.extract_scene_state()
    assert len(state) == scene.n_parts
    _check_no_penetration(scene, state)


def test_batch0_placement_does_not_mass_demote(cube_part):
    """Batch 0 is rejection-sampled against a collision manager that INCLUDES the hopper walls,
    but its spawn band is inset only by the bin wall thickness. Candidates above wall_top
    therefore overlap the hopper, exhaust the 200-attempt loop, and get demoted to a later
    wave -- silently draining the first wave and adding release waves. Placement only; no
    settle needed."""
    part, convex = cube_part
    bin_dim, n, _ = solve_bin_dim(part, fill_rate=0.2)
    n_parts = min(n, TEST_PART_CAP)
    np.random.seed(2024)          # placement is rejection-sampled; pin it for reproducibility
    scene = MujocoBinScene(part, convex, n_parts=n_parts, bin_dim=bin_dim,
                           settle_time=0.1, render=False)

    batch0 = sum(1 for b in scene._batch_of_body if b == 0)
    # An occasional miss is legitimate -- batch 0 rejection-samples up to 10 collision-free poses
    # into a small footprint. The bug was MASS demotion: 6 of 10 before the margin fix.
    assert scene._n_demoted <= max(1, 0.2 * (batch0 + scene._n_demoted)), (
        f"{scene._n_demoted} part(s) demoted out of batch 0 -- spawn band disagrees with the "
        f"hopper-inclusive collision check")


@pytest.mark.parametrize("extents,use_max_bin", [
    ((0.02, 0.02, 0.02), False),      # shrunken bin
    ((0.15, 0.15, 0.15), True),       # clips to the max bin -- exercises both ends
])
def test_pile_does_not_overspill_rim(make_box_part, extents, use_max_bin):
    """A crowned pile is expected -- BIN_TOP_MARGIN_FRAC reserves headroom and parts stack -- but
    a pile standing a full effective layer proud of the walls means height was under-provisioned.
    Part centres must also stay within the footprint: the walls are what contain them."""
    part, convex = make_box_part(extents)
    bin_dim, n, _ = solve_bin_dim(part, fill_rate=0.2)
    if use_max_bin:
        assert bin_dim == pytest.approx(MAX_BIN_DIM, abs=1e-6)

    scene = _settle(part, convex, bin_dim, min(n, TEST_PART_CAP))
    w, l, h, _ = bin_dim
    h_eff = _effective_layer_height(part)

    body_ids = scene._body_ids if scene._body_ids else list(range(1, scene.model.nbody))
    pos = scene.data.xpos[body_ids]

    z_allow = h + OVERSHOOT_TOL_LAYERS * h_eff
    assert float(pos[:, 2].max()) <= z_allow, (
        f"pile overspills the rim: max centre z={pos[:, 2].max():.4f} m > "
        f"{z_allow:.4f} m (bin height {h:.4f} + {OVERSHOOT_TOL_LAYERS}x h_eff {h_eff:.4f})")

    # Centres inside the footprint, with a quarter-radius of slack for a crowning part.
    slack = 0.25 * float(part.bounding_sphere.primitive.radius)
    assert float(np.abs(pos[:, 0]).max()) <= w / 2 + slack, "part centre outside +/-x walls"
    assert float(np.abs(pos[:, 1]).max()) <= l / 2 + slack, "part centre outside +/-y walls"
