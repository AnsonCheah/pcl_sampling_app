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


def test_warns_when_exporting_an_unrecentred_cloud(headless_app, box_mesh, capsys):
    """A bundle exported before `recenter_mesh_pcd` must not go out quietly.

    Recentring stays a deliberate user action (the Downsample button), so SaveStage cannot
    assume it happened. Without the transform the cloud is still in the raw raycast frame:
    the ambiguity axis is not a frame axis through the origin, so MechVision's
    `rotationStrategy` cannot address it, and `geo_center.json` — always written as
    identity because the frame is meant to be baked into the cloud — becomes a lie.

    `app.geocenter` is the signal: `worker()` leaves the un-applied transform there and
    `recenter_mesh_pcd` replaces it with identity.
    """
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.geocenter = np.eye(4)
    app.stage = Stage.RENDER

    # Recentred (identity) -> silent.
    app.stages[Stage.SAVE]._warn_if_not_recentred()
    assert "NOT been recentred" not in capsys.readouterr().out

    # Transform still pending -> loud.
    pending = np.eye(4)
    pending[:3, 3] = [0.01, 0.0, -0.02]
    app.geocenter = pending
    app.stages[Stage.SAVE]._warn_if_not_recentred()
    assert "NOT been recentred" in capsys.readouterr().out


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
