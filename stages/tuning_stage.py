"""tuning_stage.py — wrap the MechVision Optuna tuner into the Open3D wizard.

The last stage of the pipeline. For the part currently loaded in the wizard
(``app.mesh_basename``) it lets the operator:

1. preview / delete the synthetic scenes under ``output/synthetic_target/<part>/``,
2. launch the joint Optuna study against a live MechVision instance, with a live 3D
   overlay of the matched reference cloud that follows whichever scene the optimizer
   is evaluating (via ``MVEvaluator.on_scene_eval``) and a trial-level progress bar
   (via ``Tuner.on_trial_complete``),
3. browse the resulting Pareto configs and run any one live against the selected
   scene (coverage + time), and
4. open the Optuna dashboard (a subprocess pointed at the study's SQLite DB).

The stage is headless-drivable like the other worker stages: every GUI touch is
guarded by ``self.app.headless`` / ``app.main_thread`` (which itself no-ops headless),
so ``worker()`` runs the full study whether triggered by a button or ``_run_worker()``.

The heavy imports (optuna, mm_adapter, mesh_analysis, …) are kept local to the methods
that use them so importing this stage stays cheap and cannot break app startup.
"""

import atexit
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import colorsys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from enums import Stage
from stages.stage_base import BaseStage

# ── repo-root paths (kept off the import chain so the pure helpers stay testable) ──
_STAGES_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_STAGES_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SYNTH_ROOT  = os.path.join(_ROOT, "output", "synthetic_target")
_REFPCD_ROOT = os.path.join(_ROOT, "output", "reference_pcd")
_RESULTS_DIR = os.path.join(_ROOT, "MM_Optimizer", "results")

import MM_Optimizer.search_config as SC   # pure constants, no heavy deps

DASH_HOST = "127.0.0.1"
DASH_PORT = 8080
DASH_URL  = f"http://{DASH_HOST}:{DASH_PORT}/"

SAMPLER_DEFAULT = SC.SAMPLER_DEFAULT
SAMPLER_CHOICES = SC.SAMPLER_CHOICES

_SCENE_VOXEL = 0.002   # m — downsample the ~34 MB scene cloud for a light preview
_REF_VOXEL   = 0.004   # m — downsample the reference cloud placed at each match


# ── Windows: tie the dashboard subprocess to a Job Object so it is killed when this
# app process dies (even on a hard kill) — otherwise a killed session orphans the
# dashboard, which keeps a lock on the study .db and blocks deletion. Pure ctypes,
# no pywin32. All calls are guarded and degrade to no-op on non-Windows / failure. ──

def _make_kill_on_close_job():
    """Create a Job Object that kills its member processes when its last handle
    closes (i.e. when this process exits). Returns the job handle or None."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype  = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", ctypes.c_uint32),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", ctypes.c_uint32),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", ctypes.c_uint32),
                        ("SchedulingClass", ctypes.c_uint32)]

        class _EXT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _BASIC),
                        ("IoInfo", _IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = _EXT()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject.restype  = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                wintypes.LPVOID, wintypes.DWORD]
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def _assign_process_to_job(job, pid):
    if sys.platform != "win32" or not job:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype  = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        hproc = k32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not hproc:
            return
        k32.AssignProcessToJobObject.restype  = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject(job, hproc)
        k32.CloseHandle(hproc)
    except Exception:
        pass


def _close_handle(handle):
    if sys.platform != "win32" or not handle:
        return
    try:
        import ctypes
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
    except Exception:
        pass


def pose_to_T(pose):
    """[x, y, z, qw, qx, qy, qz] (scalar-first) -> 4x4. Mirror of visualize_match.pose_to_T."""
    x, y, z, qw, qx, qy, qz = pose
    rot = o3d.geometry.get_rotation_matrix_from_quaternion(
        np.array([qw, qx, qy, qz], dtype=float))
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3]  = (x, y, z)
    return T


def instance_color(i):
    """One colour per instance — golden-ratio hue walk at RenderStage's HSV tone
    (saturation 0.4, value 0.9) so the overlay matches the segmented-scene look."""
    return colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.4, 0.9)


class TuningStage(BaseStage):
    """Wrap MM_Optimizer/tuner.py into the GUI (see module docstring)."""

    downstream = {}   # last stage; owns no cross-stage app.* state

    def __init__(self, app):
        self.name = Stage.TUNING.name
        # Config — set BEFORE super().__init__() (it calls build_panel(), which reads them).
        self.sampler  = SAMPLER_DEFAULT
        self.n_trials = None    # None → SC default; the Run button fills these from the sliders
        self.n_rounds = None
        # Deliberate override for the model-frame guard (see _frames_agree). Off by default:
        # the check costs seconds and the run it protects costs hours.
        self.skip_frame_check = False
        # Runtime scratch (torn down in on_clear()).
        self._client       = None
        self._optimizer    = None
        self._dash_proc    = None
        self._dash_job     = None     # Windows Job handle tying the dashboard to this process
        self._atexit_hooked = False
        self._running      = False
        self._scene_cache  = {}      # scene_dir -> (scene_pcd, ref_pcd) downsampled
        self._pareto       = []      # list of (label, coarse, fine)
        self._update_pending = False # coalesce live-overlay GUI posts
        self._total_trials = 1
        self._resume_mode  = "extend"  # "restart" (fresh) | "extend" (add slider trials); headless default
        self._stop_requested = False   # Stop button → study.stop() from _on_trial_complete
        self._tuning_active  = False   # True only during the tuning study worker (gates Stop)
        super().__init__(app)

    # ─────────────────────────────────────────────────────────────────────
    # Part / path helpers (GUI-free, headless-safe, unit-tested)
    # ─────────────────────────────────────────────────────────────────────

    def _part(self):
        return self.app.mesh_basename

    def _part_dir(self):
        part = self._part()
        return os.path.join(_SYNTH_ROOT, part) if part else None

    def _db_path(self):
        part = self._part()
        return os.path.join(_RESULTS_DIR, f"{part}_{self.sampler}.db") if part else None

    def _cache_path(self):
        part = self._part()
        return os.path.join(_RESULTS_DIR, f"eval_cache_{part}.json") if part else None

    def _artifacts_exist(self):
        return any(p and os.path.exists(p) for p in (self._db_path(), self._cache_path()))

    def _scan_scenes(self):
        """Sorted scene_NNNNN directory names for the current part ([] if none)."""
        d = self._part_dir()
        if not d or not os.path.isdir(d):
            return []
        return sorted(
            name for name in os.listdir(d)
            if name.startswith("scene_") and os.path.isdir(os.path.join(d, name)))

    def _scene_sample_plys(self, scene_dir):
        return sorted(
            (os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
             if f.startswith("sample_") and f.endswith(".ply")),
            key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))

    def _delete_scene(self, scene_dir):
        """rmtree one scene directory (the Delete Scene button's real work)."""
        if scene_dir and os.path.isdir(scene_dir):
            shutil.rmtree(scene_dir)

    @staticmethod
    def _try_unlink(paths):
        """Unlink each path; return the ones that could not be removed (e.g. Windows lock)."""
        still = []
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                still.append(p)
        return still

    def _delete_tuning_artifacts(self):
        """Unlink the study DB + eval cache for the current part (reset()'s real work).

        On Windows the study .db can be locked by an orphaned optuna-dashboard from a
        killed prior session; if the first unlink fails, release the locker(s) and retry.
        Returns the list of paths that still could not be deleted."""
        targets = [p for p in (self._db_path(), self._cache_path()) if p]
        remaining = self._try_unlink(targets)
        if remaining:
            self._release_db_lockers()   # kill our dashboard + any orphaned optuna-dashboard
            for _ in range(3):
                remaining = self._try_unlink(remaining)
                if not remaining:
                    break
                time.sleep(0.3)          # give the OS a moment to release the file handle
        return remaining

    # ─────────────────────────────────────────────────────────────────────
    # Overlay / Pareto helpers (GUI-free, unit-tested)
    # ─────────────────────────────────────────────────────────────────────

    def _load_scene_clouds(self, scene_dir):
        """Return (scene_pcd, ref_pcd), downsampled and cached per scene_dir."""
        if scene_dir in self._scene_cache:
            return self._scene_cache[scene_dir]
        scene_pcd = o3d.io.read_point_cloud(os.path.join(scene_dir, "scene.ply"))
        if _SCENE_VOXEL:
            scene_pcd = scene_pcd.voxel_down_sample(_SCENE_VOXEL)
        ref_pcd = o3d.io.read_point_cloud(os.path.join(scene_dir, "reference_cloud.ply"))
        if _REF_VOXEL:
            ref_pcd = ref_pcd.voxel_down_sample(_REF_VOXEL)
        self._scene_cache[scene_dir] = (scene_pcd, ref_pcd)
        return scene_pcd, ref_pcd

    def _build_overlay(self, scene_dir, fine_poses, gt_poses=None):
        """Assemble [(name, geom, rgb), …]: grey scene cloud + one reference copy placed
        at each matched pose. Pure geometry — no SceneWidget calls (tested headlessly)."""
        scene_pcd, ref_pcd = self._load_scene_clouds(scene_dir)
        geoms = [("tuning_scene", o3d.geometry.PointCloud(scene_pcd), (0.55, 0.55, 0.55))]
        for i, pose in enumerate(fine_poses):
            inst = o3d.geometry.PointCloud(ref_pcd)
            inst.transform(pose_to_T(pose))
            geoms.append((f"tuning_match_{i}", inst, instance_color(i)))
        return geoms

    def pareto_options(self):
        """[(label, coarse, fine), …] from the finished study ([] if none)."""
        if self._optimizer is None:
            return []
        opts = []
        for i, (t, coarse, fine) in enumerate(self._optimizer.iter_pareto_configs()):
            cov, tm = t.values[0], t.values[1]
            opts.append((f"#{i}  cov={cov:.3f}  {tm:.2f}s", coarse, fine))
        return opts

    def _trial_progress(self, study, trial):
        """(fraction, label) for the progress bar. Pure — no GUI (tested headlessly)."""
        import optuna
        done = sum(1 for t in study.trials
                   if t.state != optuna.trial.TrialState.WAITING)
        total = max(1, self._total_trials)
        frac = min(done / total, 1.0)
        best = ""
        try:
            if study.best_trials:
                b = max(study.best_trials, key=lambda t: (t.values[0], -t.values[1]))
                best = f": best cov={b.values[0]:.3f}"
        except Exception:
            pass
        return frac, f"Trial {done}/{total}{best}"

    @staticmethod
    def _resolve_trial_budget(mode, n_prior, slider):
        """Round-0 trial target. 'extend' adds `slider` trials on top of the prior ones;
        'restart' (fresh, DB already deleted) targets `slider` from zero."""
        return n_prior + slider if mode == "extend" else slider

    @staticmethod
    def _eval_detail(fine_poses, gt_poses):
        """Progress-detail string for the current eval: mean translation (mm) and rotation
        (deg) over matched instances. Pure — reuses the optimizer's matcher."""
        from MM_Optimizer.mv_evaluator import match_poses_to_gt
        matched = match_poses_to_gt(list(fine_poses), list(gt_poses))
        pos = [p for p, _ in matched if p is not None]
        ang = [a for _, a in matched if a is not None]
        n_gt = len(gt_poses)
        if not pos:
            return f"no match  (0/{n_gt})"
        return (f"avg trans {np.mean(pos) * 1e3:.1f}mm  rot {np.mean(ang):.1f}°  "
                f"matched {len(pos)}/{n_gt}")

    @staticmethod
    def _repopulate(combo, items):
        """Rebuild a combobox's items while preserving the current selection when it still
        exists (Open3D resets the selection to index 0 on clear_items). Callers guard headless."""
        prev = combo.selected_text
        combo.clear_items()
        for it in items:
            combo.add_item(it)
        if prev and prev in items:
            combo.selected_text = prev

    # ─────────────────────────────────────────────────────────────────────
    # GUI panel
    # ─────────────────────────────────────────────────────────────────────

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        # --- Scene preview / manage ---
        self.combo_scene = self.register_widget(
            gui.Combobox(), lambda: len(self._scan_scenes()) > 0 and not self._running)
        self.combo_scene.set_on_selection_changed(self._on_scene_selected)
        self.btn_delete_scene = self.register_widget(
            gui.Button("Delete Scene"),
            lambda: bool(self.combo_scene.selected_text) and not self._running)
        self.btn_delete_scene.set_on_clicked(self._on_delete_scene)
        self.btn_refresh = self.register_widget(gui.Button("Refresh"),
                                                lambda: not self._running)
        self.btn_refresh.set_on_clicked(self._refresh)

        # --- Tuning controls ---
        self.slider_trials = self.register_widget(
            gui.Slider(gui.Slider.INT), lambda: not self._running)
        self.slider_trials.set_limits(5, 300)
        self.slider_trials.int_value = int(SC.N_TRIALS)
        self.slider_rounds = self.register_widget(
            gui.Slider(gui.Slider.INT), lambda: not self._running)
        self.slider_rounds.set_limits(1, 5)
        self.slider_rounds.int_value = int(SC.N_ROUNDS)
        self.combo_sampler = self.register_widget(
            gui.Combobox(), lambda: not self._running)
        for s in SAMPLER_CHOICES:
            self.combo_sampler.add_item(s)
        self.combo_sampler.selected_text = SAMPLER_DEFAULT
        self.chk_skip_frame_check = self.register_widget(
            gui.Checkbox("Skip model-frame check"), lambda: not self._running)
        self.chk_skip_frame_check.checked = self.skip_frame_check
        self.chk_skip_frame_check.set_on_checked(
            lambda c: setattr(self, "skip_frame_check", bool(c)))

        # Run/Stop toggle: "Run Tuning" when idle, "Stop" while a study is active.
        self.btn_run = self.register_widget(
            gui.Button("Run Tuning"),
            lambda: (self._tuning_active and not self._stop_requested)
                    or (len(self._scan_scenes()) > 0 and not self._running))
        self.btn_run.set_on_clicked(self._on_run_or_stop)
        self.btn_dashboard = self.register_widget(
            gui.Button("Open Dashboard"), lambda: self._dash_alive())
        self.btn_dashboard.set_on_clicked(self._open_dashboard)

        # --- Pareto configs / live run ---
        self.combo_pareto = self.register_widget(
            gui.Combobox(), lambda: len(self._pareto) > 0 and not self._running)
        self.btn_live_run = self.register_widget(
            gui.Button("Live Run (selected scene)"),
            lambda: bool(self.combo_pareto.selected_text)
                    and bool(self.combo_scene.selected_text) and not self._running)
        self.btn_live_run.set_on_clicked(self._on_live_run)
        self.lbl_result = gui.Label("")

        # --- Destructive / nav ---
        self.btn_clear = self.register_widget(
            gui.Button("Clear Tuning Result"),
            lambda: self._artifacts_exist() and not self._running)
        self.btn_clear.set_on_clicked(self._on_clear_result)
        self.btn_restart = self.register_widget(gui.Button("Restart"),
                                                lambda: not self._running)
        self.btn_restart.set_on_clicked(lambda: self.app._restart())

        v.add_child(gui.Label("MechVision Tuning"))
        v.add_child(gui.Label("Scene preview"))
        v.add_child(self.combo_scene)
        v.add_child(self.btn_delete_scene)
        v.add_child(self.btn_refresh)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Trials"))
        v.add_child(self.slider_trials)
        v.add_child(gui.Label("Rounds"))
        v.add_child(self.slider_rounds)
        v.add_child(gui.Label("Sampler"))
        v.add_child(self.combo_sampler)
        v.add_child(self.chk_skip_frame_check)
        v.add_child(self.btn_run)
        v.add_child(self.btn_dashboard)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Pareto configs"))
        v.add_child(self.combo_pareto)
        v.add_child(self.btn_live_run)
        v.add_child(self.lbl_result)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_clear)
        v.add_child(self.btn_restart)

        print("loaded tuning panel")
        return v

    def _refresh_ui(self):
        if self.app.headless:
            return
        self._populate_scene_combo()
        self._repopulate(self.combo_pareto, [label for label, _, _ in self._pareto])
        self.enable_widgets()

    def on_clear(self):
        # Non-destructive runtime teardown — runs on every state clear (restart, etc.).
        # NEVER deletes the DB/cache; only reset() (the button) does that.
        self._teardown_runtime()
        self._scene_cache = {}
        self._pareto = []
        self._running = False
        self._update_pending = False
        if not self.app.headless:
            self.combo_scene.clear_items()
            self.combo_pareto.clear_items()
            self.lbl_result.text = ""

    def reset(self):
        """Clear Tuning Result: destructive delete of the study DB + eval cache.

        Only ever reached from the Clear Tuning Result button (app._restart /
        clear_state_from call clear_downstream() directly, never reset()), so the
        file deletion cannot fire on an ordinary restart."""
        self._teardown_runtime()               # release our own handles first
        remaining = self._delete_tuning_artifacts()   # unlink db + cache (kills lockers on retry)
        self._pareto = []
        if remaining and not self.app.headless:
            self._confirm(
                "Could not delete (still in use):\n" + "\n".join(remaining) +
                "\n\nClose the Optuna dashboard browser tab and any other process "
                "using the DB, then click Clear Tuning Result again.",
                on_ok=lambda: None)
        super().reset()                        # clear_state_from → clear_downstream → on_clear; then _refresh_ui

    # ─────────────────────────────────────────────────────────────────────
    # Scene preview / management callbacks
    # ─────────────────────────────────────────────────────────────────────

    def on_enter(self):
        # On entering the stage, preview the selected (or first) on-disk scene so the 3D view
        # isn't blank/stale — important when SCENE/RENDER were skipped. Entry-only, so it never
        # clobbers a live-run / tuning overlay (those go through _refresh_ui, not on_enter).
        if self.app.headless or self._running:
            return
        names = self._scan_scenes()
        if not names:
            return
        sel = self.combo_scene.selected_text
        name = sel if sel in names else names[0]
        self._on_scene_selected(name, 0)

    def _populate_scene_combo(self):
        if self.app.headless:
            return
        self._repopulate(self.combo_scene, self._scan_scenes())

    def _refresh(self):
        # Refresh button: drop cached clouds so previews reload from disk, then one refresh path.
        self._scene_cache.clear()
        self._refresh_ui()

    def _on_scene_selected(self, text, idx):
        if self.app.headless or not text:
            return
        scene_dir = os.path.join(self._part_dir(), text)
        try:
            scene_pcd, _ = self._load_scene_clouds(scene_dir)
        except Exception as e:
            print(f"[TUNING] failed to load {scene_dir}: {e}")
            return
        mat = self._point_material([1, 1, 1])

        def apply():
            self.app._clear_scene()
            geom = o3d.geometry.PointCloud(scene_pcd)
            self.app.scene.scene.add_geometry("tuning_scene", geom, mat)
            self.app._reframe()
        self.app.main_thread(apply)
        self.enable_widgets()

    def _on_delete_scene(self):
        if self.app.headless:
            return
        name = self.combo_scene.selected_text
        if not name:
            return
        scene_dir = os.path.join(self._part_dir(), name)
        self._confirm(f"Delete {name}?\nThis permanently removes:\n{scene_dir}",
                      on_ok=lambda: (self._delete_scene(scene_dir),
                                     self._scene_cache.pop(scene_dir, None),
                                     self._refresh()))

    # ─────────────────────────────────────────────────────────────────────
    # Tuning run (heavy work — off the GUI thread)
    # ─────────────────────────────────────────────────────────────────────

    def _on_run_or_stop(self):
        # The Run button doubles as Stop while a study is active.
        if self._tuning_active:
            self._on_stop()
        else:
            self._on_run()

    def _sync_run_button(self):
        """Keep the Run/Stop toggle label in sync with the run state (main thread)."""
        if self.app.headless:
            return
        self.btn_run.text = ("Stopping..." if self._stop_requested
                             else "Stop" if self._tuning_active
                             else "Run Tuning")

    def enable_widgets(self):
        super().enable_widgets()
        self._sync_run_button()

    def _on_run(self):
        if self._running:
            return
        if not self.app.headless:
            self.n_trials = int(self.slider_trials.int_value)
            self.n_rounds = int(self.slider_rounds.int_value)
            self.sampler  = self.combo_sampler.selected_text or SAMPLER_DEFAULT
        # If a study with prior trials exists, ask whether to add to it or start fresh.
        n_prior = self._count_prior_trials()
        if n_prior > 0 and not self.app.headless:
            slider = self.n_trials if self.n_trials is not None else SC.N_TRIALS
            self.app.choice_dialog(
                f"A study for this part already has {n_prior} trials.",
                [(f"Extend +{slider}", lambda: self._launch_tuning("extend")),
                 ("Restart (fresh)",   lambda: self._launch_tuning("restart"))],
                title="Resume tuning")
        else:
            self._launch_tuning("restart")   # nothing prior → fresh (nothing to delete)

    def _launch_tuning(self, mode):
        self._resume_mode    = mode
        self._running        = True
        self._tuning_active  = True
        self._stop_requested = False
        self.start()             # BaseStage → daemon thread → worker()
        # Re-evaluate predicates now that we're running: everything but Stop/Dashboard disables.
        self.enable_widgets()

    def _count_prior_trials(self):
        db = self._db_path()
        if not db or not os.path.exists(db):
            return 0
        try:
            import optuna
            study = optuna.load_study(study_name=f"{self._part()}_{self.sampler}",
                                      storage=f"sqlite:///{db}")
            return len(study.trials)
        except Exception:
            return 0

    def _on_worker_start(self):
        self._running        = True
        self._tuning_active  = True
        self._stop_requested = False

    def _on_worker_done(self):
        self._running        = False
        self._tuning_active  = False
        self._stop_requested = False   # clear so the Run/Stop toggle returns to "Run Tuning"
        super()._on_worker_done()

    def _frames_agree(self, part, model_path) -> bool:
        """Refuse to tune when the scenes and the model are in different model frames.

        A tuning run is hours of MechVision calls scored against each scene's `T_gt`, and
        `T_gt` is only meaningful against the model frame the scene was generated in. If the
        bundle has since been re-exported into a different frame, every pose is scored
        against the wrong reference and the study optimises toward a fiction — with no error,
        because a stale scene still loads and still has a pose.

        The model frame is only reproducible while both the mesh and the sampling settings
        are unchanged: a different voxel size or view count can change which ambiguity axis
        wins, which moved the frame by 60.6 degrees and 28.3 mm on 25333MB000. And
        `bench/generate_scenes.py` is resumable, so one part can accumulate scenes from two
        sessions in two frames. Checking is a few seconds against a run measured in hours.

        Also compares the in-app cloud when the session has one, so a bundle that has drifted
        from what is currently loaded is caught as well.
        """
        import open3d as o3d

        from MM_Optimizer import model_sync
        from geometry.geom_utils import reference_frames_agree

        if self.skip_frame_check:
            print("[TUNING] Model-frame check skipped by request.")
            return True

        problems = model_sync.check_scene_frames(part, model_path, _SYNTH_ROOT)
        live = self.app.down_pcd_surface
        if live is not None:
            why = reference_frames_agree(live, o3d.io.read_point_cloud(model_path))
            if why:
                problems.append(f"{model_path} (exported bundle) vs the cloud loaded in the app"
                                f"\n        {why}")
        if not problems:
            return True

        print("[TUNING] " + "=" * 62)
        print(f"[TUNING] ABORTING: {len(problems)} model-frame disagreement(s) for '{part}'.")
        print("[TUNING] Every pose would be scored against the wrong reference frame.")
        for p in problems:
            print(f"[TUNING]   {p}")
        print(f"[TUNING] Fix: python bench/generate_scenes.py --only {part} --force")
        print("[TUNING] Or tick 'Skip model-frame check' to run anyway.")
        print("[TUNING] " + "=" * 62)
        return False

    def worker(self):
        # Local heavy imports (see module docstring).
        from MM_Optimizer.mv_evaluator    import PROJ_NAME, RESULTS_DIR, ENABLE_CACHE
        from MM_Optimizer.optimizer_utils import list_synthetic_scenes
        from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
        from MM_Optimizer.eval_cache      import EvalCache
        from MM_Optimizer.tuner import Tuner
        from mm_adapter.mm_adapter        import MechVisionClient
        import optuna

        part = self._part()
        if not part:
            print("[TUNING] No part loaded (app.mesh_basename unset).")
            return

        scenes_root  = os.path.join(_SYNTH_ROOT, part)
        scene_groups = (list_synthetic_scenes(scenes_root) if os.path.isdir(scenes_root) else [])
        if not scene_groups:
            print(f"[TUNING] No scenes under {scenes_root}.")
            return

        model_path = os.path.join(_REFPCD_ROOT, part, f"{part}_surface", f"{part}_surface.ply")
        if not os.path.exists(model_path):
            print(f"[TUNING] Reference model not found: {model_path} "
                  f"— run the sampling pipeline first.")
            return

        if not self._frames_agree(part, model_path):
            return

        # Per-run scene budget (SC globals are mutated the same way the CLI does).
        SC.M_FULL  = len(scene_groups)

        slider   = self.n_trials if self.n_trials is not None else SC.N_TRIALS
        n_rounds = self.n_rounds if self.n_rounds is not None else SC.N_ROUNDS

        os.makedirs(RESULTS_DIR, exist_ok=True)
        if self._resume_mode == "restart":
            self._delete_tuning_artifacts()   # DB + eval cache → fresh study

        # Pre-create/load the study so the dashboard has a schema immediately, and to read the
        # prior trial count (0 after a restart) — needed for the extend budget and the progress
        # total. run() attaches to the same study via load_if_exists=True.
        n_prior = 0
        try:
            _pre = optuna.create_study(
                study_name = f"{part}_{self.sampler}",
                storage    = f"sqlite:///{self._db_path()}",
                directions = ["maximize", "minimize"],
                load_if_exists = True)
            n_prior = len(_pre.trials)
        except Exception as e:
            print(f"[TUNING] could not pre-create study DB: {e}")

        # Round-0 budget: 'extend' adds `slider` on top of prior trials; 'restart' targets `slider`.
        # `done` in _trial_progress counts ALL trials, so the progress total must use this budget.
        budget = self._resolve_trial_budget(self._resume_mode, n_prior, slider)
        self._total_trials = max(
            1, budget + (n_rounds - 1) * SC.N_TRIALS_REFINE)

        # Connect to MechVision (handled failure, not a crash).
        try:
            client   = MechVisionClient()
            projects = client.get_projects()
        except Exception as e:
            print(f"[TUNING] MechVision unavailable: {e}")
            return
        if PROJ_NAME not in projects:
            print(f"[TUNING] Project '{PROJ_NAME}' not found in MechVision.")
            try:
                client.close()
            except Exception:
                pass
            return
        project_id  = projects[PROJ_NAME]
        self._client = client

        pcd = load_reference_pcd(model_path)
        ws  = analyze_mesh(pcd)
        cache = EvalCache(self._cache_path(), enabled=ENABLE_CACHE)

        self._optimizer = Tuner(
            part_name      = part,
            client         = client,
            project_id     = project_id,
            scene_groups   = scene_groups,
            warm_start     = ws,
            cache          = cache,
            dry_run        = False,
            n_trials = budget,
            n_rounds       = n_rounds,
            storage_path   = os.path.join(RESULTS_DIR, ""),
            sampler        = self.sampler,
        )
        self._optimizer.on_scene_eval = self._on_scene_eval
        self._optimizer.on_trial_complete = self._on_trial_complete

        self._launch_dashboard()

        if not self.app.headless:
            self.app.show_progress("Tuning.....")

        try:
            result = self._optimizer.run()
        except Exception as e:
            print(f"[TUNING] Optimization failed: {e}")
            result = None

        if result is not None:
            try:
                out = self._optimizer.export_best(result, prefix=f"{self.sampler.upper()}_")
                print(f"[TUNING] Best config exported -> {out}")
            except Exception as e:
                print(f"[TUNING] export_best failed: {e}")

        # Publish the Pareto front.
        self._pareto = self.pareto_options()
        print(f"[TUNING] Pareto front: {len(self._pareto)} configs")
        if not self.app.headless:
            def fill():
                self._repopulate(self.combo_pareto, [label for label, _, _ in self._pareto])
                self.enable_widgets()
            self.app.main_thread(fill)

    # ─────────────────────────────────────────────────────────────────────
    # Live overlay + trial progress (fired from the optimizer thread)
    # ─────────────────────────────────────────────────────────────────────

    def _on_scene_eval(self, scene_dir, coarse, fine, fine_poses, gt_poses):
        if self.app.headless:
            return
        # Always refresh the progress detail (cheap), even when the 3D frame is coalesced away.
        try:
            self.app.update_progress(detail=self._eval_detail(fine_poses, gt_poses))
        except Exception:
            pass
        if self._update_pending:
            return   # coalesce: drop this overlay frame if a prior GUI update is still pending
        self._update_pending = True
        try:
            geoms = self._build_overlay(scene_dir, list(fine_poses), gt_poses)
        except Exception as e:
            print(f"[TUNING] overlay build failed: {e}")
            self._update_pending = False
            return

        def apply():
            self.app._clear_scene()
            for name, geom, rgb in geoms:
                self.app.scene.scene.add_geometry(name, geom, self._point_material(rgb))
            self.app.redraw()
            self._update_pending = False
        self.app.main_thread(apply)

    def _on_trial_complete(self, study, trial):
        if self._stop_requested:
            study.stop()   # graceful: optuna ends the optimize loop after this trial finishes
        if self.app.headless:
            return
        frac, text = self._trial_progress(study, trial)
        self.app.update_progress(frac, text)

    def _on_stop(self):
        self._stop_requested = True
        self.app.update_progress(text="Stopping after current trial…")
        self.enable_widgets()   # re-label the toggle to "Stopping…" and disable it

    # ─────────────────────────────────────────────────────────────────────
    # Pareto live run (selected scene only)
    # ─────────────────────────────────────────────────────────────────────

    def _on_live_run(self):
        if self.app.headless or self._running:
            return
        self._running = True
        self.app.show_progress("Live run starting")
        # Assign to worker_thread so the base nav-lock (BaseStage.nav_enabled) covers it too.
        self.worker_thread = threading.Thread(target=self._live_run_worker, daemon=True)
        self.worker_thread.start()
        self.enable_widgets()   # re-evaluate predicates (Stop/Dashboard) + lock Back/Next

    def _live_run_worker(self):
        try:
            self._do_live_run()
        finally:
            self._running = False
            self.app.main_thread(self._on_worker_done)

    def _do_live_run(self):
        if self._optimizer is None:
            print("[TUNING] No optimizer, run tuning first.")
            return
        idx        = self.combo_pareto.selected_index
        scene_name = self.combo_scene.selected_text
        if idx is None or idx < 0 or idx >= len(self._pareto):
            print("[TUNING] No Pareto config selected.")
            return
        if not scene_name:
            print("[TUNING] No scene selected.")
            return
        _, coarse, fine = self._pareto[idx]
        scene_dir = os.path.join(self._part_dir(), scene_name)
        plys = self._scene_sample_plys(scene_dir)
        if not plys:
            print(f"[TUNING] No sample_*.ply in {scene_dir}.")
            return

        ev = self._optimizer
        _, gt_poses = ev._prepare_scene(plys)   # (scene_dir, gt_poses), cached
        res = ev._run_one_scene(coarse, fine, scene_dir, gt_poses,
                                SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT)  # fires _on_scene_eval
        cov = res["instance_coverage"]
        tm  = res["coarse_time_s"] + res["fine_time_s"]
        text = f"cov={cov:.3f}   time={tm:.2f}s"
        print(f"[TUNING] Live run {scene_name}: {text}")
        if not self.app.headless:
            self.app.main_thread(lambda: setattr(self.lbl_result, "text", text))

    # ─────────────────────────────────────────────────────────────────────
    # Optuna dashboard subprocess
    # ─────────────────────────────────────────────────────────────────────

    def _dashboard_exe(self):
        d = os.path.dirname(sys.executable)
        for cand in (os.path.join(d, "Scripts", "optuna-dashboard.exe"),
                     os.path.join(d, "optuna-dashboard.exe"),
                     os.path.join(d, "optuna-dashboard")):
            if os.path.exists(cand):
                return cand
        return shutil.which("optuna-dashboard")

    def _dash_alive(self):
        return self._dash_proc is not None and self._dash_proc.poll() is None

    def _launch_dashboard(self):
        if self.app.headless:
            return   # dashboard is a GUI companion; headless would leak the server process
        if self._dash_alive():
            return
        db = self._db_path()
        exe = self._dashboard_exe()
        if not db or exe is None:
            print("[TUNING] optuna-dashboard executable not found -- skipping dashboard.")
            return
        try:
            self._dash_proc = subprocess.Popen(
                [exe, f"sqlite:///{db}", "--host", DASH_HOST, "--port", str(DASH_PORT)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Tie it to a kill-on-close Job so it dies with this app even on a hard kill
            # (prevents the orphaned-dashboard DB lock). atexit covers clean window close.
            self._dash_job = _make_kill_on_close_job()
            _assign_process_to_job(self._dash_job, self._dash_proc.pid)
            if not self._atexit_hooked:
                atexit.register(self._teardown_runtime)
                self._atexit_hooked = True
            print(f"[TUNING] optuna-dashboard -> {DASH_URL}  (db={db})")
        except Exception as e:
            print(f"[TUNING] failed to launch dashboard: {e}")
            self._dash_proc = None
        if not self.app.headless:
            self.app.main_thread(self.enable_widgets)

    def _open_dashboard(self):
        import webbrowser
        webbrowser.open(DASH_URL)

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle helpers
    # ─────────────────────────────────────────────────────────────────────

    def _teardown_runtime(self):
        """Terminate the dashboard subprocess + close the MechVision client. Idempotent."""
        if self._dash_proc is not None:
            try:
                self._dash_proc.terminate()
            except Exception:
                pass
            self._dash_proc = None
        if self._dash_job is not None:
            _close_handle(self._dash_job)   # closing the last job handle kills any survivor
            self._dash_job = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _release_db_lockers(self):
        """Best-effort release of whatever holds the study DB: our own dashboard, plus
        orphaned optuna-dashboard processes left by a killed prior session (which a fresh
        app instance has no handle to). On Windows the orphan is killed by image name —
        safe here because the stage runs at most one dashboard (fixed port)."""
        self._teardown_runtime()
        if sys.platform == "win32":
            # /T kills the whole tree: the optuna-dashboard.exe launcher AND the child
            # python.exe that actually serves the app and holds the SQLite lock.
            try:
                subprocess.run(["taskkill", "/F", "/T", "/IM", "optuna-dashboard.exe"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10)
            except Exception:
                pass

    def _on_clear_result(self):
        self._confirm(
            "Delete the tuning study DB and eval cache for this part?\n"
            f"{self._db_path()}\n{self._cache_path()}",
            on_ok=self.reset)

    def _point_material(self, rgb):
        # Mirror RenderStage's point-cloud material (render_stage.py:300-302) so the
        # overlay looks consistent with the other stages: point_size 1.5, base_color =
        # rgb, and the default shader (RenderStage leaves the shader unset).
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.point_size = 1.5
        mat.base_color = list(rgb) + [1.0]
        return mat

    def _confirm(self, message, on_ok):
        """Modal OK/Cancel confirm — delegates to the shared app dialog."""
        self.app.confirm_dialog(message, on_ok)
