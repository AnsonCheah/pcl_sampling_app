"""
test_scene_disk_preview.py — fast tests for the SceneStage skip + disk-scene mesh preview
------------------------------------------------------------------------------------------
Run:  python -m pytest stages/tests/test_scene_disk_preview.py -q

Headless, no MuJoCo/physics — temp dirs + tiny meshes + a synthetic scene_state.npz.
(The physics smoke test lives in test_scene_stage.py, which is module-level `slow`.)
"""

import os

import numpy as np
import pytest

from enums import Stage
from geometry.file_utils import list_scene_dirs
import stages.scene_stage as ss


@pytest.fixture
def scene(headless_app, box_mesh):
    headless_app.mesh_basename = "PART"
    headless_app.target_mesh = box_mesh
    return headless_app.stages[Stage.SCENE]


# ── directory scanning ──────────────────────────────────────────────────────

def test_list_scene_dirs_sorted(tmp_path):
    for name in ("scene_00002", "scene_00000", "scene_00001"):
        (tmp_path / name).mkdir()
    (tmp_path / "notascene").mkdir()
    (tmp_path / "scene_x.txt").write_text("x")
    assert list_scene_dirs(tmp_path) == ["scene_00000", "scene_00001", "scene_00002"]


def test_list_scene_dirs_missing():
    assert list_scene_dirs("/does/not/exist") == []


# ── bin reconstruction ──────────────────────────────────────────────────────

def test_reconstruct_bin_mesh_nonempty():
    mesh = ss._reconstruct_bin_mesh([0.4, 0.3, 0.25, 0.01], np.eye(4))
    assert mesh.has_vertices()
    assert len(mesh.vertices) == 40    # floor + 4 walls, 8 verts each


# ── skip via Next ───────────────────────────────────────────────────────────

def test_next_enabled_with_disk_scenes(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(scene, "_synth_dir", lambda: tmp_path)
    assert len(scene.app.o3d_scene) == 0
    assert scene.next_enabled() is False           # nothing generated, nothing on disk
    (tmp_path / "scene_00000").mkdir()
    assert scene.next_enabled() is True            # on-disk scene → advance allowed


def test_request_next_skip_confirms_and_proceeds(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(scene, "_synth_dir", lambda: tmp_path)
    (tmp_path / "scene_00000").mkdir()
    called = []
    # headless confirm_dialog runs on_ok immediately → proceed fires.
    scene.request_next(lambda: called.append(1))
    assert called == [1]


# ── disk-scene mesh reconstruction ──────────────────────────────────────────

def _write_scene_state(scene_dir, translations):
    scene_dir.mkdir(parents=True, exist_ok=True)
    n = len(translations)
    T_gt = np.tile(np.eye(4), (n, 1, 1))
    for i, t in enumerate(translations):
        T_gt[i, :3, 3] = t
    np.savez(scene_dir / "scene_state.npz",
             T_gt=T_gt,
             bin_dim=np.array([0.4, 0.3, 0.25, 0.01]),
             bin_transform=np.eye(4))


def test_build_disk_scene_geoms_places_parts(scene, tmp_path):
    translations = [[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]]
    scene_dir = tmp_path / "scene_00000"
    _write_scene_state(scene_dir, translations)

    geoms = scene._build_disk_scene_geoms(str(scene_dir))

    assert [name for name, _ in geoms] == ["disk_part_0", "disk_part_1", "disk_bin"]
    box_centroid = np.asarray(scene.app.target_mesh.get_center())
    for i, t in enumerate(translations):
        moved = np.asarray(geoms[i][1].get_center())
        assert np.allclose(moved, box_centroid + np.array(t), atol=1e-6)


def test_build_disk_scene_geoms_missing_npz(scene, tmp_path):
    (tmp_path / "scene_00000").mkdir()
    assert scene._build_disk_scene_geoms(str(tmp_path / "scene_00000")) == []
