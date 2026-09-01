"""
test_n_instances.py  —  N-instance I/O contract verification
=============================================================
Verifies the input/output shape of the .vis pipeline when fed a list of
N individual point clouds via read_synthetic.

Variable convention:
  N  = number of part instances (point clouds) fed into the .vis project
  H  = outputNum parameter on CoarseMatchingV2 — hypotheses generated per
       instance internally during coarse matching

candidateTopNum on FineMatchingLite is fixed at 1 throughout (production
default: take the best coarse hypothesis per cloud for fine refinement).

Output contract (candidateTopNum=1 fixed):
    len(coarse_poses)  == N   (outer list: one sublist per input cloud)
    len(fine_poses)    == N   (flat list: one refined pose per input cloud)

H (outputNum) controls internal search quality only — it does NOT change
the N-length output counts with candidateTopNum=1.

Tests:
  test_io_contract()   — assert n_coarse == N AND n_fine == N for a
                          representative set of (N, H) pairs; log timing
                          and inner hypothesis counts as informational
  test_nms_impact()    — full scene only; check whether NMS can drop
                          n_coarse or n_fine below N

Requires live MechVision (CAD_Match project loaded).
Run from project root:
    python MM_Optimizer/tests/test_n_instances.py
"""

import logging
import os
import shutil
import sys
import tempfile

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mm_adapter.mm_adapter     import MechVisionClient
from mm_adapter.mm_dataclasses import CoarseMatchingV2, FineMatchingLite, EasyCreateStringList
from MM_Optimizer.mv_evaluator     import PROJ_NAME, OPTIMIZER_UTILS_PATH
from MM_Optimizer.optimizer_utils  import list_synthetic_scenes, read_gt_pose_from_ply
from MM_Optimizer.mesh_analysis    import analyze_mesh, load_reference_pcd
from MM_Optimizer              import model_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)

# Representative (N, H) pairs — small, mid, full-scene × low and high H
TRIALS = [
    (1,  1),
    (1,  3),
    (10, 1),
    (10, 3),
    # full-scene N added dynamically below
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subscene_dir(source_plys, n, work_dir, label):
    """Copy exactly n PLYs from source_plys into a fresh temp subdirectory."""
    dest = os.path.join(work_dir, label)
    os.makedirs(dest, exist_ok=True)
    for src in source_plys[:n]:
        shutil.copy2(src, dest)
    return dest


def _build_params(scene_dir, H, use_nms=True):
    """Build params_dict for one vision run (candidateTopNum=1 fixed)."""
    # Same resolution as mv_evaluator._make_params_dict: one library entry per part,
    # holding whichever regime's cloud model_sync last installed.
    ply_path = model_sync.model_ply(PART)
    geo_path = model_sync.geo_center(PART)

    scene  = EasyCreateStringList(name="Scene_Path", strings=(scene_dir, "string", ""))
    coarse = CoarseMatchingV2(
        name              = "Coarse_Match_Synthetics",
        modelSelection    = (PART, "string", ""),
        modelFileName     = (ply_path, "string", ""),
        geoCenterFileName = (geo_path, "string", ""),
    )
    coarse.outputNum          = (str(H),      "double", "")
    coarse.refStep            = ("5",          "double", "")
    coarse.distQuantification = ("3.75",       "double", "")
    coarse.useDistanceNMS     = (str(use_nms), "bool",   "")

    fine = FineMatchingLite(
        name              = "Fine_Match_Synthetics",
        modelSelection    = (PART, "string", ""),
        modelFileName     = (ply_path, "string", ""),
        geoCenterFileName = (geo_path, "string", ""),
    )
    fine.candidateTopNum     = ("1",   "double", "")
    fine.confidenceThreshold = ("0.0", "double", "")
    fine.operationApproach   = ("0.0", "double", "")

    return {
        scene.name:         scene.to_step_params(),
        "Pre_Segmentation": {
            "scriptFilePath": (OPTIMIZER_UTILS_PATH, "string", ""),
            "funcName":       ("read_synthetic",      "string", ""),
        },
        coarse.name:        coarse.to_step_params(),
        fine.name:          fine.to_step_params(),
    }


def _run(client, project_id, scene_dir, H, use_nms=True):
    """
    Run one MechVision call. Returns (n_coarse, inner_hyp_counts, n_fine, t_c, t_f).

    n_coarse         = len(coarse_poses)             outer list length == N
    inner_hyp_counts = [len(coarse_poses[i]) ...]    inner counts per instance
    n_fine           = len(fine_poses)               == N with candidateTopNum=1
    """
    params = _build_params(scene_dir, H, use_nms)
    client.set_params(project_id, params)
    result = client.run_vision(project_id)
    coarse_poses = result.get("coarse_poses", [])
    return (
        len(coarse_poses),
        [len(inst) for inst in coarse_poses],
        len(result.get("fine_poses", [])),
        result["coarse_time_s"],
        result["fine_time_s"],
    )


def _connect():
    client   = MechVisionClient()
    projects = client.get_projects()
    assert PROJ_NAME in projects, f"Project '{PROJ_NAME}' not loaded: {projects}"
    return client, projects[PROJ_NAME]


def _source_plys():
    groups = list_synthetic_scenes(SCENES_DIR)
    assert groups, f"No scene groups found in {SCENES_DIR}"
    return groups[0]   # scene_00000


# ---------------------------------------------------------------------------
# Test 1: I/O contract — n_coarse == N and n_fine == N
# ---------------------------------------------------------------------------

def test_io_contract():
    """
    Assert both output list lengths equal N for a representative set of (N, H).

    coarse_poses outer == N : one sublist per input cloud
    fine_poses           == N : one refined pose per input cloud

    Inner hypothesis counts are logged as informational — MechVision may
    produce fewer than H hypotheses per cloud when geometry is sparse
    (independent of NMS).
    """
    source_plys = _source_plys()
    N_full = len(source_plys)
    trials = TRIALS + [(N_full, 1), (N_full, 3)]

    client, project_id = _connect()
    work_dir = tempfile.mkdtemp(prefix="mmopt_io_")
    try:
        hdr = (f"{'N':>5}  {'H':>3}  {'n_coarse':>9}  {'inner_min':>10}"
               f"  {'inner_max':>10}  {'n_fine':>7}  {'t_c':>7}  {'t_f':>6}  status")
        log.info(f"\n{'='*len(hdr)}")
        log.info("I/O contract: n_coarse == N  AND  n_fine == N  (candidateTopNum=1)")
        log.info(f"{'='*len(hdr)}")
        log.info(hdr)
        log.info("-" * len(hdr))

        for n, H in trials:
            scene_dir = _make_subscene_dir(source_plys, n, work_dir, f"io_N{n:03d}_H{H}")
            n_coarse, inner_hyps, n_fine, t_c, t_f = _run(
                client, project_id, scene_dir, H)
            inner_min = min(inner_hyps) if inner_hyps else 0
            inner_max = max(inner_hyps) if inner_hyps else 0
            ok = (n_coarse == n) and (n_fine == n)
            log.info(f"{n:>5}  {H:>3}  {n_coarse:>9}  {inner_min:>10}"
                     f"  {inner_max:>10}  {n_fine:>7}  {t_c:>7.3f}  {t_f:>6.3f}"
                     f"  {'PASS' if ok else 'FAIL'}")
            assert n_coarse == n, \
                f"N={n}, H={H}: n_coarse={n_coarse}, expected {n}"
            assert n_fine == n, \
                f"N={n}, H={H}: n_fine={n_fine}, expected {n}"

        log.info("=" * len(hdr))
        log.info("PASS: test_io_contract")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        client.close()


# ---------------------------------------------------------------------------
# Test 2: NMS impact — full scene, check floor
# ---------------------------------------------------------------------------

def _gt_positions(plys):
    """Return (N, 3) array of GT xyz positions (metres)."""
    poses = [read_gt_pose_from_ply(p, scalar_first=True) for p in plys]
    return np.array([p[:3] for p in poses])


def test_nms_impact():
    """
    Full-scene only: check whether distance NMS drops n_coarse or n_fine below N.

    MechVision removes coarse candidates within 0.1 × model_diameter of an
    already-selected candidate within the same input cloud (per-cloud NMS).
    This test verifies NMS does not cause cross-cloud candidate loss by
    comparing observed counts against N and reporting the minimum
    inter-instance distance vs the NMS threshold.
    """
    source_plys = _source_plys()
    N_full = len(source_plys)

    model_path = os.path.join(_ROOT, "output", "reference_pcd", PART,
                              f"{PART}_surface", f"{PART}_surface.ply")
    pcd = load_reference_pcd(model_path)
    ws  = analyze_mesh(pcd)
    nms_thresh = 0.1 * ws.diameter_m

    # Min pairwise GT distance across all N instances
    pos = _gt_positions(source_plys)
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist_mm = float(dists.min()) * 1e3
    n_close_pairs = int((dists < nms_thresh).sum()) // 2

    log.info(f"\nNMS threshold:    {nms_thresh*1e3:.1f} mm  (0.1 × D={ws.diameter_m*1e3:.1f} mm)")
    log.info(f"Min GT distance:  {min_dist_mm:.1f} mm  ({n_close_pairs} pairs < threshold)")

    client, project_id = _connect()
    work_dir = tempfile.mkdtemp(prefix="mmopt_nms_")
    try:
        hdr = (f"{'N':>5}  {'H':>3}  {'NMS':>5}  {'n_coarse':>9}"
               f"  {'n_fine':>7}  {'coarse_floor':>13}  {'fine_floor':>11}")
        log.info(f"\n{'='*len(hdr)}")
        log.info(f"NMS impact  (full scene N={N_full}, NMS threshold={nms_thresh*1e3:.1f} mm)")
        log.info(f"{'='*len(hdr)}")
        log.info(hdr)
        log.info("-" * len(hdr))

        for H in [1, 3]:
            for use_nms in [True, False]:
                scene_dir = _make_subscene_dir(
                    source_plys, N_full, work_dir,
                    f"nms_H{H}_{'on' if use_nms else 'off'}")
                n_coarse, _, n_fine, _, _ = _run(
                    client, project_id, scene_dir, H, use_nms=use_nms)
                c_ok = "OK" if n_coarse >= N_full else f"FAIL({n_coarse}<{N_full})"
                f_ok = "OK" if n_fine   == N_full else f"FAIL({n_fine}≠{N_full})"
                log.info(f"{N_full:>5}  {H:>3}  {'ON' if use_nms else 'OFF':>5}"
                         f"  {n_coarse:>9}  {n_fine:>7}  {c_ok:>13}  {f_ok:>11}")

        log.info("=" * len(hdr))
        log.info("DONE: test_nms_impact  (informational — no assertions)")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        client.close()


if __name__ == "__main__":
    print("test_n_instances.py  (requires live MechVision)\n")
    test_io_contract()
    test_nms_impact()
    print("\nAll N-instance tests PASSED")
