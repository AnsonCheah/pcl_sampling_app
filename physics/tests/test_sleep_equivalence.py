"""Gate for MuJoCo's sleeping-islands flag (mjENBL_SLEEP).

OUTCOME: the flag is OFF. It was measured, it failed this gate, and it bought nothing.

  - Speed, 48-part random scene: 27.3 s with sleeping vs 28.0 s without, and ZERO trees asleep
    at the end. The batched release keeps the pile moving and simulate() exits as soon as
    is_settled() trips, so there is no quiescent tail for sleeping to exploit. (The 32x figure
    from the MuJoCo benchmark comes from stepping long after a pile has settled.)
  - Fidelity: settled positions moved by up to 4.5 mm on a 20 mm part -- 22% of the part.

So it costs pose fidelity and returns no speedup for the random-arrangement pipeline. The
equivalence test below is skipped rather than deleted: it is the gate to re-run if anyone wants
to enable sleeping for a scene type with a long static tail (structured partition/tray can run
the full settle_time). Do NOT widen the tolerances to make it pass -- re-measure instead.
"""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from physics.mujoco_bin_scene import MujocoBinScene, solve_bin_dim

pytestmark = pytest.mark.slow

SEED = 12345
N_PARTS = 16
POS_TOL_FRAC = 0.10     # position agreement, as a fraction of the part's bounding-sphere radius
ANG_TOL_DEG = 10.0      # orientation agreement


def _settle(part, convex, bin_dim, enable_sleep):
    """Identical seed => identical spawn poses, so any divergence is the flag's doing."""
    np.random.seed(SEED)
    scene = MujocoBinScene(part, convex, n_parts=N_PARTS, bin_dim=bin_dim,
                           render=False, enable_sleep=enable_sleep)
    scene.simulate()
    return scene, scene.extract_scene_state()


def test_sleep_is_disabled_by_default(cube_part):
    """The shipped configuration must not enable sleeping -- see the module docstring."""
    part, convex = cube_part
    bin_dim, _, _ = solve_bin_dim(part, fill_rate=0.2)
    scene = MujocoBinScene(part, convex, n_parts=2, bin_dim=bin_dim, settle_time=0.1,
                           render=False)
    assert scene.enable_sleep is False
    assert not (scene.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP))


@pytest.mark.skip(reason="sleeping is off (no speedup, perturbs poses) -- re-run this gate "
                         "before enabling it for any scene type; see module docstring")
def test_sleep_preserves_settled_poses(cube_part):
    part, convex = cube_part
    bin_dim, _, _ = solve_bin_dim(part, fill_rate=0.2)

    _, state_off = _settle(part, convex, bin_dim, enable_sleep=False)
    _, state_on = _settle(part, convex, bin_dim, enable_sleep=True)

    assert set(state_off) == set(state_on)

    r_bs = float(part.bounding_sphere.primitive.radius)
    pos_tol = POS_TOL_FRAC * r_bs

    dpos, dang = [], []
    for name in state_off:
        dpos.append(float(np.linalg.norm(
            np.asarray(state_on[name]["position"]) - np.asarray(state_off[name]["position"]))))
        q_off = R.from_quat(state_off[name]["quaternion"], scalar_first=True)
        q_on = R.from_quat(state_on[name]["quaternion"], scalar_first=True)
        dang.append(float(np.degrees((q_off.inv() * q_on).magnitude())))

    max_dpos, max_dang = max(dpos), max(dang)
    print(f"[SLEEP] max position delta {max_dpos * 1000:.3f} mm (tol {pos_tol * 1000:.3f} mm), "
          f"max orientation delta {max_dang:.2f} deg (tol {ANG_TOL_DEG} deg)")

    assert max_dpos <= pos_tol, (
        f"sleeping changed settled positions by up to {max_dpos * 1000:.2f} mm "
        f"(> {pos_tol * 1000:.2f} mm) -- do not ship the flag")
    assert max_dang <= ANG_TOL_DEG, (
        f"sleeping changed settled orientations by up to {max_dang:.1f} deg -- do not ship")


@pytest.mark.skip(reason="companion to the skipped equivalence gate above; on the random "
                         "pipeline it fails -- zero trees sleep, which is why the flag is off")
def test_sleep_actually_sleeps(cube_part):
    """Guards against a silent no-op: if nothing ever sleeps, the flag buys nothing and the
    equivalence test above would pass vacuously."""
    part, convex = cube_part
    bin_dim, _, _ = solve_bin_dim(part, fill_rate=0.2)
    scene, _ = _settle(part, convex, bin_dim, enable_sleep=True)

    assert hasattr(scene.data, "tree_asleep"), "mjData.tree_asleep missing -- API changed"
    n_asleep = int(np.sum(np.asarray(scene.data.tree_asleep) >= 0))
    print(f"[SLEEP] {n_asleep} tree(s) asleep at end of settle, ncon={scene.data.ncon}")
    assert n_asleep > 0, "no tree ever slept -- the flag is not doing anything"
