"""Tests for SaveStage: the load-bearing PLY geocenter comment keys (RENDER branch) and
the reference-bundle file output (SAVE branch)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from enums import Stage

# Keys consumed by downstream loaders — must never silently disappear (see root CLAUDE.md).
REQUIRED_COMMENT_KEYS = [
    "geocenter_x", "geocenter_y", "geocenter_z",
    "geocenter_qx", "geocenter_qy", "geocenter_qz", "geocenter_qw",
]


def test_render_branch_writes_geocenter_comments(headless_app, box_mesh, tmp_path):
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.geocenter = np.eye(4)
    app.stage = Stage.RENDER

    app.stages[Stage.SAVE].worker(path=tmp_path)

    ply = tmp_path / "reference_cloud.ply"
    assert ply.exists()
    header = ply.read_text(errors="ignore")
    for key in REQUIRED_COMMENT_KEYS:
        assert key in header, f"missing PLY comment key: {key}"


def _profile_with_axis(direction, point):
    from geometry.ambiguity import AmbiguityAxis, AmbiguityProfile
    ax = AmbiguityAxis(direction=np.asarray(direction, float), point=np.asarray(point, float),
                       fold=2, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    return AmbiguityProfile(axes=[ax], dominant=ax)


def test_warns_when_the_ambiguity_axis_is_not_addressable(headless_app, box_mesh, capsys):
    """A bundle whose ambiguity axis is not a frame axis through the origin must not go out
    quietly.

    Recentring stays a deliberate user action (the Downsample button), so SaveStage cannot
    assume it happened. `rotationStrategy` can only rotate about a geocenter frame axis
    through the origin, and `geo_center.json` is always written as identity — so if the frame
    is not the cloud's own, the symmetry search sweeps the wrong line at any angleStep.

    This asks the geometry rather than `app.geocenter`, which now records what has been
    APPLIED rather than what is pending — the old check would have fired on exactly the
    recentred bundles it was meant to bless.
    """
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.stage = Stage.RENDER
    save = app.stages[Stage.SAVE]

    # Axis is frame Z through the origin -> silent.
    app.ambiguity_profile = _profile_with_axis([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    save._warn_if_axis_not_addressable()
    assert "cannot address it" not in capsys.readouterr().out

    # Right direction, but the axis misses the origin -> loud.
    app.ambiguity_profile = _profile_with_axis([0.0, 0.0, 1.0], [0.004, 0.0, 0.0])
    save._warn_if_axis_not_addressable()
    assert "cannot address it" in capsys.readouterr().out

    # Through the origin, but not a frame axis -> loud.
    app.ambiguity_profile = _profile_with_axis([0.6, 0.0, 0.8], [0.0, 0.0, 0.0])
    save._warn_if_axis_not_addressable()
    assert "cannot address it" in capsys.readouterr().out

    # No ambiguity axis at all -> nothing to be wrong about.
    app.ambiguity_profile = None
    save._warn_if_axis_not_addressable()
    assert "cannot address it" not in capsys.readouterr().out


def test_geocenter_comments_carry_the_applied_frame(headless_app, box_mesh, tmp_path):
    """The values, not just the keys.

    These were `0.0` in every PLY ever exported: the old code read `geocenter[3, 0..2]`, but
    `pcd_geocenter` returns `inv([R|o])` whose row 3 is always `[0,0,0,1]`. Asserting only
    that the keys are present (as the test below does) could never catch that.
    """
    from scipy.spatial.transform import Rotation as R

    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.stage = Stage.RENDER

    # A frame whose origin sat at (11, -23, 7) mm before recentring, rotated about Z.
    origin = np.array([0.011, -0.023, 0.007])
    rot = R.from_euler("z", 30.0, degrees=True).as_matrix()
    applied = np.eye(4)
    applied[:3, :3] = rot.T
    applied[:3, 3] = -rot.T @ origin
    app.geocenter = applied

    app.stages[Stage.SAVE].worker(path=tmp_path)
    header = (tmp_path / "reference_cloud.ply").read_text(errors="ignore")
    got = {k: float(v) for k, v in
           (ln.split()[1:3] for ln in header.splitlines()
            if ln.startswith("comment geocenter_"))}

    assert np.allclose([got["geocenter_x"], got["geocenter_y"], got["geocenter_z"]],
                       origin, atol=1e-9), "translation must be the model origin, not zeros"
    assert np.allclose([got["geocenter_qx"], got["geocenter_qy"],
                        got["geocenter_qz"], got["geocenter_qw"]],
                       R.from_matrix(rot).as_quat(), atol=1e-9)


def test_geocenter_comments_are_identity_when_nothing_was_applied(headless_app, box_mesh, tmp_path):
    """An un-recentred bundle reports identity — honest rather than uninformative."""
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.geocenter = np.eye(4)
    app.stage = Stage.RENDER

    app.stages[Stage.SAVE].worker(path=tmp_path)
    header = (tmp_path / "reference_cloud.ply").read_text(errors="ignore")
    got = {k: float(v) for k, v in
           (ln.split()[1:3] for ln in header.splitlines()
            if ln.startswith("comment geocenter_"))}
    assert np.allclose([got["geocenter_x"], got["geocenter_y"], got["geocenter_z"]], 0.0)
    assert np.isclose(got["geocenter_qw"], 1.0)


@pytest.mark.slow
def test_save_branch_writes_reference_bundle(sampled_app):
    app = sampled_app
    app.mesh_basename = "pytest_throwaway_part"
    app.stage = Stage.SAVE
    base = Path(__file__).resolve().parent.parent.parent / "output" / "reference_pcd" / app.mesh_basename
    try:
        app.stages[Stage.SAVE].worker()
        assert (base / f"{app.mesh_basename}_surface" / f"{app.mesh_basename}_surface.ply").exists()
        assert (base / f"{app.mesh_basename}.stl").exists()
        assert app.output_pcd_path is not None
    finally:
        shutil.rmtree(base, ignore_errors=True)
