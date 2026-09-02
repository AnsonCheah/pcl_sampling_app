"""Tests for SaveStage: the load-bearing PLY geocenter comment keys (RENDER branch) and
the reference-bundle file output (SAVE branch)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from enums import Stage

# Keys consumed by downstream loaders -- must never silently disappear (see root CLAUDE.md).
REQUIRED_COMMENT_KEYS = [
    "geocenter_x", "geocenter_y", "geocenter_z",
    "geocenter_qx", "geocenter_qy", "geocenter_qz", "geocenter_qw",
    "ambiguity_fold", "ambiguity_aligned",
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


def _profile_with_axis(direction, point, fold=2, frame_changed=False):
    from geometry.ambiguity import AmbiguityAxis, AmbiguityProfile
    ax = AmbiguityAxis(direction=np.asarray(direction, float), point=np.asarray(point, float),
                       fold=fold, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    return AmbiguityProfile(axes=[ax], dominant=ax, frame_changed=frame_changed)


def test_warns_when_the_ambiguity_axis_is_not_addressable(headless_app, box_mesh, capsys):
    """A bundle whose ambiguity axis is not a frame axis through the origin must not go out
    quietly.

    Recentring stays a deliberate user action (the Downsample button), so SaveStage cannot
    assume it happened. `rotationStrategy` can only rotate about a geocenter frame axis
    through the origin, and `geo_center.json` is always written as identity -- so if the frame
    is not the cloud's own, the symmetry search sweeps the wrong line at any angleStep.

    This asks the geometry rather than `app.geocenter`, which now records what has been
    APPLIED rather than what is pending -- the old check would have fired on exactly the
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
    """An un-recentred bundle reports identity -- honest rather than uninformative."""
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


# -----------------------------------------------------------------------------
# Ambiguity metadata -- what lets the tuner aim MechVision's symmetry search
# -----------------------------------------------------------------------------
#
# `ambiguity_fold` uses the crystallographic convention, so no negative sentinel:
#   0 = continuous (body of revolution)   1 = C1, no rotational symmetry   N = N-fold
# `ambiguity_aligned` says whether frame Z really IS the ambiguity axis, which is the
# precondition for setting rotationStrategy=Z.

def _ambiguity_comments(tmp_path, app):
    app.stages[Stage.SAVE].worker(path=tmp_path)
    header = (tmp_path / "reference_cloud.ply").read_text(errors="ignore")
    return {k: int(v) for k, v in
            (ln.split()[1:3] for ln in header.splitlines()
             if ln.startswith("comment ambiguity_"))}


def test_ambiguity_comments_present_when_recentred(headless_app, box_mesh, tmp_path):
    """A bundle recentred to the ambiguity axis advertises its fold and says so."""
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.stage = Stage.RENDER
    app.ambiguity_profile = _profile_with_axis([0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                                               fold=4, frame_changed=True)

    got = _ambiguity_comments(tmp_path, app)
    assert got["ambiguity_fold"] == 4
    assert got["ambiguity_aligned"] == 1


def test_ambiguity_aligned_is_zero_for_pca_frame(headless_app, box_mesh, tmp_path):
    """An axis was found but the user never clicked Recenter, so the frame is PCA.

    This is exactly the case a geocenter-quaternion check cannot catch: a PCA recentre
    also leaves a non-identity quaternion, so 'quaternion != identity' would wrongly bless
    it and the symmetry search would sweep an arbitrary line.
    """
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.stage = Stage.RENDER
    app.ambiguity_profile = _profile_with_axis([0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                                               fold=4, frame_changed=False)

    got = _ambiguity_comments(tmp_path, app)
    assert got["ambiguity_fold"] == 4, "the fold is still known even when unaligned"
    assert got["ambiguity_aligned"] == 0


def test_asymmetric_part_writes_fold_one(headless_app, box_mesh, tmp_path):
    """No axis at all is C1 -- fold 1, not a missing key.

    The keys are always written so a reader can distinguish 'analysed, found nothing' from
    'exported before these keys existed'.
    """
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.stage = Stage.RENDER
    app.ambiguity_profile = None

    got = _ambiguity_comments(tmp_path, app)
    assert got["ambiguity_fold"] == 1
    assert got["ambiguity_aligned"] == 0


def test_continuous_part_writes_fold_zero(headless_app, box_mesh, tmp_path):
    """A body of revolution is fold 0 -- the tuner scores those position-only."""
    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.stage = Stage.RENDER
    app.ambiguity_profile = _profile_with_axis([0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                                               fold=0, frame_changed=True)

    got = _ambiguity_comments(tmp_path, app)
    assert got["ambiguity_fold"] == 0
    assert got["ambiguity_aligned"] == 1


def test_aligned_is_zero_when_frame_changed_but_no_axis(headless_app, box_mesh, tmp_path):
    """The `frame_changed` without `dominant` trap.

    If the ambiguity checkbox is on but no axis was found, `pcd_geocenter(pcd, axis=None)`
    silently falls through to the PCA branch while `_geocenter_for` still sets
    `frame_changed = True`. Trusting that flag alone would advertise a PCA frame as
    ambiguity-aligned, and the tuner would sweep the wrong line.
    """
    from geometry.ambiguity import AmbiguityProfile

    app = headless_app
    app.down_pcd = box_mesh.sample_points_uniformly(500)
    app.stage = Stage.RENDER
    app.ambiguity_profile = AmbiguityProfile(axes=[], dominant=None, frame_changed=True)

    got = _ambiguity_comments(tmp_path, app)
    assert got["ambiguity_fold"] == 1
    assert got["ambiguity_aligned"] == 0, "a PCA frame must never report as aligned"


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
