"""Headless guards for the unified camera reframe and the origin-axes toggle.

The camera maths, the glyph font, and the widget layout all need a live renderer and are covered
by the manual pass. What IS testable headlessly is the thing most likely to break: the headless
early-returns. Every stage now calls `app._reframe`, so a missing guard, a bad attribute, or a
stale import would take down the whole pipeline in batch mode.
"""

import numpy as np
import pytest
from open3d.geometry import AxisAlignedBoundingBox

from app import MeshSamplingApp, DEFAULT_WORLD_EXTENT, FRAME_REL_TOL
from enums import Stage


def _bbox(lo, hi):
    return AxisAlignedBoundingBox(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))


# ---------------------------------------------------------------------------
# Content-change detection: the camera must hold still when a stage re-shows
# the same object, and must move when the viewport really changes.
# ---------------------------------------------------------------------------

def test_framing_changed_on_first_call(headless_app):
    assert headless_app._framed_bbox is None
    assert headless_app._framing_changed(_bbox((0, 0, 0), (0.2, 0.1, 0.2)))


def test_identical_bbox_is_unchanged(headless_app):
    b = _bbox((0, 0, 0), (0.2, 0.1, 0.2))
    headless_app._framed_bbox = ((0.1, 0.05, 0.1), float(np.linalg.norm([0.2, 0.1, 0.2])))
    assert not headless_app._framing_changed(b)


def test_cloud_sampled_from_a_mesh_counts_as_the_same_object(headless_app):
    """Raycast shows raw_pcd where Import showed the mesh — the sampled cloud misses the mesh
    extremes by a hair. That must not be treated as new content, or navigating between the two
    stages throws away the user's orbit."""
    mesh = _bbox((0, 0, 0), (0.2, 0.1, 0.2))
    headless_app._remember_framing(mesh)
    cloud = _bbox((0.001, 0.0005, 0.001), (0.199, 0.0995, 0.199))
    assert not headless_app._framing_changed(cloud)


def test_part_to_bin_counts_as_new_content(headless_app):
    """Generating a scene swaps a 0.2 m part for a 0.76 m bin — the camera must follow."""
    headless_app._remember_framing(_bbox((0, 0, 0), (0.2, 0.1, 0.2)))
    assert headless_app._framing_changed(_bbox((-0.38, -0.29, 0.0), (0.38, 0.29, 0.25)))


def test_same_size_but_moved_counts_as_new_content(headless_app):
    headless_app._remember_framing(_bbox((0, 0, 0), (0.2, 0.1, 0.2)))
    assert headless_app._framing_changed(_bbox((1.0, 0, 0), (1.2, 0.1, 0.2)))


def test_tolerance_is_relative_to_size(headless_app):
    """The same *proportional* change must give the same verdict on a 5 cm part and a 5 m one."""
    for scale in (0.05, 5.0):
        headless_app._framed_bbox = None
        headless_app._remember_framing(_bbox((0, 0, 0), (scale, scale, scale)))
        nudge = 0.5 * FRAME_REL_TOL * scale          # comfortably inside tolerance
        assert not headless_app._framing_changed(
            _bbox((nudge, nudge, nudge), (scale + nudge, scale + nudge, scale + nudge)))
        big = 3.0 * FRAME_REL_TOL * scale            # comfortably outside
        assert headless_app._framing_changed(
            _bbox((big, big, big), (scale + big, scale + big, scale + big)))


def test_redraw_is_a_noop_headless(headless_app):
    headless_app.redraw()   # must not touch self.scene / self.window


def test_default_world_bbox_is_framable():
    """The empty-scene fallback must be a real, non-degenerate box — _reframe divides by its
    diagonal, so a degenerate one would reintroduce the unresponsive-R symptom."""
    bbox = MeshSamplingApp._default_world_bbox()
    assert not bbox.is_empty()
    extent = np.asarray(bbox.get_extent())
    assert np.linalg.norm(extent) > 1e-9
    assert np.isclose(extent.max(), DEFAULT_WORLD_EXTENT * 1.15)


def test_default_world_bbox_contains_the_origin():
    """It frames the world axes, which are drawn from the origin outward, so the origin has to be
    inside the box or the axes' base sits off-screen."""
    bbox = MeshSamplingApp._default_world_bbox()
    lo = np.asarray(bbox.get_min_bound())
    hi = np.asarray(bbox.get_max_bound())
    assert np.all(lo <= 0) and np.all(hi >= 0)
    # Asymmetric: axes run origin -> +extent, so most of the box is on the positive side.
    assert np.all(hi > -lo)


def test_reframe_is_a_noop_headless(headless_app):
    # No window, no renderer — must return before touching self.scene.
    assert headless_app.target_mesh is None
    headless_app._reframe()          # must not raise


def test_reframe_noop_with_a_mesh_loaded(headless_app, box_mesh):
    """The old target_mesh fallback is gone; loading a mesh must not tempt _reframe into
    touching the (nonexistent) renderer."""
    headless_app.target_mesh = box_mesh
    headless_app._reframe()          # must not raise


def test_reframe_takes_no_bbox_or_view_kwargs(headless_app):
    """_reframe_scene and the bbox/view overrides were removed — callers must not resurrect them."""
    assert not hasattr(headless_app, "_reframe_scene")
    with pytest.raises(TypeError):
        headless_app._reframe(bbox=object())
    with pytest.raises(TypeError):
        headless_app._reframe(view=(0, 0, 0))


def test_every_stage_refresh_ui_is_headless_safe(headless_app, box_mesh):
    """Cheap regression net for the reframe calls just added across the stages: a typo'd
    attribute or missing self.app. prefix would raise here."""
    headless_app.target_mesh = box_mesh
    for stage, inst in headless_app.stages.items():
        inst._refresh_ui()           # every stage returns early when headless


def test_headless_app_constructs_without_touching_gui():
    """Guards where _install_glyph_font sits: it must stay inside the `not headless` branch,
    since gui.Application.set_font requires an initialised GUI app."""
    app = MeshSamplingApp(headless=True)
    assert app.headless
    assert not hasattr(app, "btn_axes")
    assert not hasattr(app, "axes_glyph")


def test_glyph_fallback_is_ascii():
    """If the label ever falls back it must still render — the point of the fallback is that a
    missing glyph draws a blank button."""
    from app import MeshSamplingApp as A
    import inspect
    src = inspect.getsource(A._install_glyph_font)
    assert '"XYZ"' in src
