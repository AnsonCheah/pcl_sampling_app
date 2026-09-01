"""
test_tuning_stage.py — headless tests for the TUNING stage's button custom functions
-------------------------------------------------------------------------------------
Run:  python -m pytest stages/tests/test_tuning_stage.py -q

Every check runs headless (MeshSamplingApp(headless=True)) and touches no live
MechVision — temp dirs, tiny point clouds, and in-memory optuna studies stand in.
Module-level path constants (_SYNTH_ROOT, _RESULTS_DIR) are monkeypatched to tmp
dirs so nothing under the real output/ tree is created or deleted.
"""

import os

import numpy as np
import open3d as o3d
import optuna
import pytest

from enums import Stage
import stages.tuning_stage as ts


@pytest.fixture
def tuning(headless_app):
    """The TUNING stage from a headless app, with a part name set."""
    headless_app.mesh_basename = "PART"
    return headless_app.stages[Stage.TUNING]


# ── scene enumeration / deletion ────────────────────────────────────────────

def test_scan_scenes_sorted(tuning, tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_SYNTH_ROOT", str(tmp_path))
    part_dir = tmp_path / "PART"
    for name in ("scene_00002", "scene_00000", "scene_00001"):
        (part_dir / name).mkdir(parents=True)
    (part_dir / "not_a_scene").mkdir()
    (part_dir / "scene_00003.txt").write_text("x")   # file, not a dir
    assert tuning._scan_scenes() == ["scene_00000", "scene_00001", "scene_00002"]


def test_scan_scenes_no_part(headless_app, tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_SYNTH_ROOT", str(tmp_path))
    headless_app.mesh_basename = None
    assert headless_app.stages[Stage.TUNING]._scan_scenes() == []


def test_delete_scene(tuning, tmp_path):
    scene = tmp_path / "scene_00000"
    (scene / "sub").mkdir(parents=True)
    (scene / "scene.ply").write_text("x")
    tuning._delete_scene(str(scene))
    assert not scene.exists()


# ── tuning-artifact deletion (Clear Tuning Result) ──────────────────────────

def _write_artifacts(tuning):
    db = tuning._db_path()
    cache = tuning._cache_path()
    os.makedirs(os.path.dirname(db), exist_ok=True)
    with open(db, "w") as f:
        f.write("db")
    with open(cache, "w") as f:
        f.write("{}")
    return db, cache


def test_delete_tuning_artifacts(tuning, tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_RESULTS_DIR", str(tmp_path))
    db, cache = _write_artifacts(tuning)
    assert tuning._artifacts_exist()
    tuning._delete_tuning_artifacts()
    assert not os.path.exists(db) and not os.path.exists(cache)
    assert not tuning._artifacts_exist()


def test_ordinary_clear_keeps_artifacts(tuning, tmp_path, monkeypatch):
    """clear_downstream() (an ordinary state clear) must NOT delete the DB/cache —
    only reset() (the Clear button) does."""
    monkeypatch.setattr(ts, "_RESULTS_DIR", str(tmp_path))
    db, cache = _write_artifacts(tuning)
    tuning.clear_downstream()   # runs on_clear() -> _teardown_runtime, not file deletion
    assert os.path.exists(db) and os.path.exists(cache)


def test_reset_deletes_artifacts(tuning, tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_RESULTS_DIR", str(tmp_path))
    db, cache = _write_artifacts(tuning)
    tuning.reset()   # override: teardown -> delete -> super().reset()
    assert not os.path.exists(db) and not os.path.exists(cache)


def test_delete_happy_path_does_not_kill_lockers(tuning, tmp_path, monkeypatch):
    """No lock → unlink succeeds first try; the blunt locker-kill is never invoked."""
    monkeypatch.setattr(ts, "_RESULTS_DIR", str(tmp_path))
    called = []
    monkeypatch.setattr(tuning, "_release_db_lockers", lambda: called.append(1))
    _write_artifacts(tuning)
    assert tuning._delete_tuning_artifacts() == []
    assert called == [], "locker kill must not run when nothing is locked"


def test_delete_retries_after_releasing_lockers(tuning, monkeypatch):
    """A locked DB → _release_db_lockers() is called, then the retry succeeds."""
    monkeypatch.setattr(ts.time, "sleep", lambda *_: None)
    calls = {"unlink": 0, "release": 0}

    def fake_unlink(paths):
        calls["unlink"] += 1
        return list(paths) if calls["unlink"] == 1 else []   # locked once, freed after release

    monkeypatch.setattr(tuning, "_try_unlink", fake_unlink)
    monkeypatch.setattr(tuning, "_release_db_lockers",
                        lambda: calls.__setitem__("release", calls["release"] + 1))

    remaining = tuning._delete_tuning_artifacts()
    assert remaining == []
    assert calls["release"] == 1
    assert calls["unlink"] >= 2


# ── Pareto options ──────────────────────────────────────────────────────────

def test_pareto_options_empty_without_optimizer(tuning):
    assert tuning._optimizer is None
    assert tuning.pareto_options() == []


def test_pareto_options_from_stub_optimizer(tuning):
    """pareto_options() labels + maps each front entry to (coarse, fine)."""
    t0 = optuna.trial.create_trial(params={}, distributions={}, values=[0.9, 1.2])
    t1 = optuna.trial.create_trial(params={}, distributions={}, values=[0.8, 0.4])

    class _StubOpt:
        def iter_pareto_configs(self):
            return [(t0, {"c": 0}, {"f": 0}), (t1, {"c": 1}, {"f": 1})]

    tuning._optimizer = _StubOpt()
    opts = tuning.pareto_options()
    assert [o[0] for o in opts] == ["#0  cov=0.900  1.20s", "#1  cov=0.800  0.40s"]
    assert opts[0][1] == {"c": 0} and opts[1][2] == {"f": 1}


# ── overlay assembly ────────────────────────────────────────────────────────

def _write_scene(scene_dir, n=8):
    scene_dir = str(scene_dir)
    os.makedirs(scene_dir, exist_ok=True)
    pts = np.random.RandomState(0).rand(n, 3).astype(np.float64)
    scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(os.path.join(scene_dir, "scene.ply"), scene)
    ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts * 0.1))
    o3d.io.write_point_cloud(os.path.join(scene_dir, "reference_cloud.ply"), ref)
    return scene_dir, np.asarray(ref.points)


def test_build_overlay_one_ref_per_pose(tuning, tmp_path, monkeypatch):
    # No downsampling so tiny clouds survive and centroids are exact.
    monkeypatch.setattr(ts, "_SCENE_VOXEL", 0.0)
    monkeypatch.setattr(ts, "_REF_VOXEL", 0.0)
    scene_dir, ref_pts = _write_scene(tmp_path / "scene_00000")
    ref_centroid = ref_pts.mean(axis=0)

    poses = [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],   # pure translation
             [-1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]
    geoms = tuning._build_overlay(scene_dir, poses)

    assert len(geoms) == 1 + len(poses)
    assert geoms[0][0] == "tuning_scene"
    assert [g[0] for g in geoms[1:]] == ["tuning_match_0", "tuning_match_1"]
    # Each match is the reference translated by its pose.
    for i, pose in enumerate(poses):
        moved = np.asarray(geoms[1 + i][1].points).mean(axis=0)
        assert np.allclose(moved, ref_centroid + np.array(pose[:3]), atol=1e-6)
    # rgb triples returned per geom.
    assert all(len(g[2]) == 3 for g in geoms)


# ── trial progress ──────────────────────────────────────────────────────────

def test_trial_progress(tuning):
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.add_trial(optuna.trial.create_trial(params={}, distributions={}, values=[0.9, 1.0]))
    study.add_trial(optuna.trial.create_trial(params={}, distributions={}, values=[0.7, 0.5]))
    tuning._total_trials = 10

    frac, text = tuning._trial_progress(study, study.trials[-1])
    assert abs(frac - 0.2) < 1e-9        # 2 non-waiting / 10
    assert text.startswith("Trial 2/10")
    assert "best cov=0.900" in text      # max coverage among the front


def test_trial_progress_caps_at_one(tuning):
    study = optuna.create_study(directions=["maximize", "minimize"])
    for _ in range(5):
        study.add_trial(optuna.trial.create_trial(params={}, distributions={}, values=[0.5, 1.0]))
    tuning._total_trials = 2   # fewer than actual → fraction must clamp to 1.0
    frac, _ = tuning._trial_progress(study, study.trials[-1])
    assert frac == 1.0


# ── resume budget / progress total ──────────────────────────────────────────

def test_resolve_trial_budget():
    assert ts.TuningStage._resolve_trial_budget("restart", 200, 50) == 50
    assert ts.TuningStage._resolve_trial_budget("extend", 200, 50) == 250
    assert ts.TuningStage._resolve_trial_budget("extend", 0, 50) == 50   # fresh via extend


def test_extended_total_drives_progress(tuning):
    """Extend +50 on a 200-trial study: total=250, so 201 done reads Trial 201/250 (~80%),
    not the old 201/50 overflow."""
    budget = ts.TuningStage._resolve_trial_budget("extend", n_prior=200, slider=50)
    tuning._total_trials = max(1, budget)   # n_rounds=1 → no refine term
    study = optuna.create_study(directions=["maximize", "minimize"])
    for _ in range(201):
        study.add_trial(optuna.trial.create_trial(params={}, distributions={}, values=[0.5, 1.0]))
    frac, text = tuning._trial_progress(study, study.trials[-1])
    assert text.startswith("Trial 201/250")
    assert 0.79 < frac < 0.81


# ── current-eval detail line ────────────────────────────────────────────────

def test_eval_detail_perfect_match(tuning):
    poses = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
             [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    detail = tuning._eval_detail(poses, poses)      # identical → 0 error, all matched
    assert "avg trans 0.0mm" in detail
    assert "rot 0.0°" in detail
    assert "matched 2/2" in detail


def test_eval_detail_no_match(tuning):
    gt   = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    far  = [[9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]]   # well beyond POS_THRESH_MATCH
    detail = tuning._eval_detail(far, gt)
    assert detail.startswith("no match")
    assert "0/1" in detail


# ── Stop toggle → study.stop() ──────────────────────────────────────────────

class _StopStudy:
    def __init__(self):
        self.stopped = False
        self.trials = []
        self.best_trials = []
    def stop(self):
        self.stopped = True


def test_stop_flag_calls_study_stop(tuning):
    study = _StopStudy()
    tuning._stop_requested = False
    tuning._on_trial_complete(study, None)
    assert not study.stopped              # no stop requested → study keeps running
    tuning._stop_requested = True
    tuning._on_trial_complete(study, None)
    assert study.stopped                  # Stop requested → graceful study.stop()


def test_worker_done_clears_run_state(tuning):
    """After a (stopped) run finishes, the flags reset so the Run/Stop toggle returns to
    'Run Tuning' instead of getting stuck on 'Stopping…'."""
    tuning._running = tuning._tuning_active = tuning._stop_requested = True
    tuning._on_worker_done()
    assert tuning._running is False
    assert tuning._tuning_active is False
    assert tuning._stop_requested is False


# ── selection-preserving repopulate ─────────────────────────────────────────

class _StubCombo:
    def __init__(self):
        self.items = []
        self.selected_text = ""
    def clear_items(self):
        self.items = []
        self.selected_text = ""      # Open3D resets selection on clear
    def add_item(self, name):
        self.items.append(name)
        if not self.selected_text:
            self.selected_text = name


def test_repopulate_preserves_selection():
    combo = _StubCombo()
    for n in ("a", "b", "c"):
        combo.add_item(n)
    combo.selected_text = "b"
    ts.TuningStage._repopulate(combo, ["a", "b", "c"])
    assert combo.selected_text == "b"    # still present → preserved


def test_repopulate_resets_when_gone():
    combo = _StubCombo()
    for n in ("a", "b", "c"):
        combo.add_item(n)
    combo.selected_text = "z"            # no longer in the new list
    ts.TuningStage._repopulate(combo, ["a", "b", "c"])
    assert combo.selected_text == "a"    # falls back to default (first item)


# ── navigation lock while a worker runs (general BaseStage behaviour) ─────────

class _FakeThread:
    def __init__(self, alive):
        self._alive = alive
    def is_alive(self):
        return self._alive


def test_nav_enabled_locks_while_worker_runs(tuning):
    tuning.worker_thread = None
    assert tuning.nav_enabled() is True            # idle → nav free
    tuning.worker_thread = _FakeThread(alive=True)
    assert tuning.nav_enabled() is False           # worker running → nav locked
    tuning.worker_thread = _FakeThread(alive=False)
    assert tuning.nav_enabled() is True            # worker finished → nav free


# ── model-frame staleness guard ──────────────────────────────────────────────

def _ref_cloud(n=600, seed=0, shift=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.05, 0.05, (n, 3)) + np.asarray(shift, dtype=float)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (n, 1)))
    return pcd


def _bundle_and_scenes(tmp_path, monkeypatch, scene_shifts):
    """A model PLY plus one scene per entry in `scene_shifts` (0.0 == same frame)."""
    monkeypatch.setattr(ts, "_SYNTH_ROOT", str(tmp_path / "synth"))
    model = _ref_cloud()
    model_path = tmp_path / "PART_surface.ply"
    o3d.io.write_point_cloud(str(model_path), model)
    for i, dx in enumerate(scene_shifts):
        d = tmp_path / "synth" / "PART" / f"scene_{i:05d}"
        d.mkdir(parents=True)
        o3d.io.write_point_cloud(str(d / "reference_cloud.ply"), _ref_cloud(shift=(dx, 0.0, 0.0)))
    return str(model_path)


def test_frames_agree_passes_when_every_scene_matches(tuning, tmp_path, monkeypatch):
    model_path = _bundle_and_scenes(tmp_path, monkeypatch, [0.0, 0.0, 0.0])
    assert tuning._frames_agree("PART", model_path) is True


def test_frames_agree_blocks_a_scene_from_another_frame(tuning, tmp_path, monkeypatch, capsys):
    """A tuning run is hours of MechVision calls scored against each scene's `T_gt`, which is
    only meaningful against the frame that scene was generated in. `bench/generate_scenes.py`
    is resumable, so one part really can accumulate scenes from two frames."""
    model_path = _bundle_and_scenes(tmp_path, monkeypatch, [0.0, 0.02, 0.0])
    assert tuning._frames_agree("PART", model_path) is False
    out = capsys.readouterr().out
    assert "ABORTING" in out and "scene_00001" in out
    assert "scene_00000" not in out, "only the offending scene should be named"


def test_frames_agree_can_be_overridden(tuning, tmp_path, monkeypatch, capsys):
    model_path = _bundle_and_scenes(tmp_path, monkeypatch, [0.02])
    tuning.skip_frame_check = True
    assert tuning._frames_agree("PART", model_path) is True
    assert "skipped by request" in capsys.readouterr().out


def test_frames_agree_checks_the_live_app_cloud(tuning, tmp_path, monkeypatch):
    """Catches a bundle that has drifted from the cloud currently loaded in the session."""
    model_path = _bundle_and_scenes(tmp_path, monkeypatch, [0.0])
    tuning.app.down_pcd_surface = _ref_cloud()
    assert tuning._frames_agree("PART", model_path) is True

    tuning.app.down_pcd_surface = _ref_cloud(shift=(0.02, 0.0, 0.0))
    assert tuning._frames_agree("PART", model_path) is False
