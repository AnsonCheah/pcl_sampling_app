"""Tests for the declarative downstream-clearing mechanism.

Covers `BaseStage.downstream` / `clear_downstream`, `app.clear_state_from`,
`app.reset_all_state`, and `app._restart()`. These are the highest-value tests: they pin
the single-source-of-truth contract that replaced the hand-written per-stage clears.
"""

import numpy as np

from enums import Stage


# Every app.* attribute owned by some stage, with the default it must reset to.
# Mirrors the union of all stage `downstream` maps — a parity guard against drift.
OWNED_DEFAULTS = {
    "target_mesh": None,
    "mesh_basename": None,
    "raw_pcd": None,
    "cropped_pcd": None,
    "point_count_mean": None,
    "point_count_range": None,
    "down_pcd": None,
    "down_pcd_surface": None,
    "down_pcd_edge": None,
    "feature_pcd": None,
    "pcd_flat": None,
    "geocenter": "EYE4",
    "output_pcd_path": None,
    "convex_meshes": [],
    "o3d_scene": {},
    "mj_scene": None,
    "scene_mesh": None,
    "synthetic_targets": {},
    "synthetic_scenes": {},
}


def _seed_sentinels(app):
    """Put a non-default marker on every owned attribute."""
    for attr in OWNED_DEFAULTS:
        setattr(app, attr, f"SENTINEL_{attr}")


def _is_default(app, attr):
    val = getattr(app, attr)
    expected = OWNED_DEFAULTS[attr]
    if expected == "EYE4":
        return isinstance(val, np.ndarray) and np.allclose(val, np.eye(4))
    return val == expected


def test_clear_state_from_strict_order(headless_app):
    app = headless_app
    _seed_sentinels(app)

    app.clear_state_from(Stage.DOWNSAMPLE)

    # DOWNSAMPLE and everything later is reset...
    for attr in ("down_pcd", "down_pcd_surface", "geocenter", "output_pcd_path",
                 "convex_meshes", "o3d_scene", "mj_scene", "synthetic_targets",
                 "synthetic_scenes"):
        assert _is_default(app, attr), f"{attr} should be reset"
    # ...while strictly-upstream products survive.
    assert app.target_mesh == "SENTINEL_target_mesh"
    assert app.raw_pcd == "SENTINEL_raw_pcd"
    assert app.cropped_pcd == "SENTINEL_cropped_pcd"
    assert app.point_count_range == "SENTINEL_point_count_range"


def test_strict_order_over_clears_convex_meshes(headless_app):
    # Documents the accepted consequence of strict-order clearing: DECOMPOSE sits after
    # RAYCAST in enum order, so a RAYCAST reset wipes convex_meshes even though
    # decomposition only depends on the mesh. Safe (recomputed), just not minimal.
    app = headless_app
    app.convex_meshes = [1, 2, 3]
    app.clear_state_from(Stage.RAYCAST)
    assert app.convex_meshes == []


def test_clear_state_from_exclusive(headless_app):
    app = headless_app
    app.down_pcd = "KEEP"
    app.output_pcd_path = "WIPE"
    app.convex_meshes = [1]

    app.clear_state_from(Stage.DOWNSAMPLE, inclusive=False)

    assert app.down_pcd == "KEEP"          # the stage's own product is preserved
    assert app.output_pcd_path is None      # SAVE and later are cleared
    assert app.convex_meshes == []


def test_restart_resets_all_owned_state(headless_app):
    app = headless_app
    _seed_sentinels(app)

    app._restart()

    for attr in OWNED_DEFAULTS:
        assert _is_default(app, attr), f"{attr} not at default after _restart"
    assert app.stage == Stage.IMPORT_MESH


def test_defaults_are_fresh_objects(headless_app):
    # Factories (not shared literals) must yield distinct mutable objects each clear,
    # otherwise two resets would alias the same dict/list/array.
    app = headless_app
    app.clear_state_from(Stage.SCENE)
    first_scene, first_meshes = app.o3d_scene, app.convex_meshes
    app.clear_state_from(Stage.DECOMPOSE)
    assert app.o3d_scene is not first_scene
    assert app.convex_meshes is not first_meshes
    app.geocenter[0, 0] = 99.0
    app.clear_state_from(Stage.DOWNSAMPLE)
    assert app.geocenter[0, 0] == 1.0       # fresh identity, not the mutated one


def test_scene_render_reset_headless_no_crash(headless_app):
    # Regression: reset() used to touch combobox widgets that never exist in headless.
    app = headless_app
    app.stages[Stage.SCENE].reset()
    app.stages[Stage.RENDER].reset()
    assert app.synthetic_targets == {}
    assert app.synthetic_scenes == {}
    assert app.o3d_scene == {}
    assert app.mj_scene is None


def test_every_owned_attr_declared_once(headless_app):
    # The union of all stage `downstream` maps must exactly equal OWNED_DEFAULTS, and no
    # attribute may be declared by two stages (ambiguous ownership).
    app = headless_app
    seen = {}
    for st, inst in app.stages.items():
        for attr in inst.downstream:
            assert attr not in seen, f"{attr} owned by both {seen[attr]} and {st}"
            seen[attr] = st
    assert set(seen) == set(OWNED_DEFAULTS)
