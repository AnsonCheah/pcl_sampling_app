"""
Tests for the simulate() on_step callback that drives the live GUI mesh preview.

The callback runs inside simulate()'s stepping loop (the worker thread), so these tests
verify two contracts the GUI preview relies on:
  1. on_step is invoked during stepping (at least once, roughly total_steps/preview_tick).
  2. An on_step that raises does NOT abort the sim — physics still runs to completion and
     settles. This is what insulates the physics from a GUI hiccup.

Run (in the `autotune` env):
  python -m pytest physics/tests/test_on_step_callback.py -q
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics.mujoco_bin_scene import MujocoBinScene
from physics.tests.test_bin_scene import generate_test_mesh


def _make_scene(n_parts=4):
    part_mesh, convex_meshes = generate_test_mesh("cube")
    # Short settle so the test is fast; render=False (no passive viewer).
    return MujocoBinScene(
        part_mesh, convex_meshes,
        n_parts=n_parts, render=False, arrangement="random", settle_time=1.0,
    )


def test_on_step_invoked_during_sim():
    scene = _make_scene()
    calls = {"n": 0}

    def cb():
        calls["n"] += 1

    scene.simulate(on_step=cb, preview_interval_s=0.05)
    # At least the batch-0 phase + final settle fire the callback several times.
    assert calls["n"] > 0, "on_step was never called during simulate()"


def test_on_step_exception_does_not_abort_sim():
    scene = _make_scene()
    calls = {"n": 0}

    def bad_cb():
        calls["n"] += 1
        raise RuntimeError("simulated GUI hiccup")

    # Must not propagate; sim must complete and produce a valid settled scene state.
    scene.simulate(on_step=bad_cb, preview_interval_s=0.05)
    assert calls["n"] > 0, "on_step was never called"

    state = scene.extract_scene_state()
    assert len(state) == len(scene.scene_objects), "sim did not complete after on_step raised"


def test_on_step_default_none_is_noop():
    # The default path (no callback) must behave exactly as before.
    scene = _make_scene()
    scene.simulate()
    state = scene.extract_scene_state()
    assert len(state) == len(scene.scene_objects)


if __name__ == "__main__":
    test_on_step_invoked_during_sim()
    test_on_step_exception_does_not_abort_sim()
    test_on_step_default_none_is_noop()
    print("All on_step callback tests passed.")
