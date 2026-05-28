"""
ppf_auto_params.py
------------------
Derives all PPF + Hough voting parameters from observable model geometry.
All tunables are expressed as multipliers so grid search over synthetics
can optimise them without per-part manual work.
"""

import numpy as np
import open3d as o3d


# ── Multiplier defaults (optimise these via synthetic grid search) ────────────
# Expressed relative to model diameter D or surface area SA.

DEFAULTS = dict(
    n_model_target       = 3000,
    scene_count_cap_k    = 3.0,
    relative_dist_step   = 0.045,
    num_angles           = 60,
    # match() params — named to match OpenCV directly
    scene_sample_step    = 5,      # relativeSceneSampleStep = 1/scene_sample_step
                                   # higher = faster, less accurate. 5 → 0.2
    # relativeSceneDistance mirrors relativeSamplingStep from training
    # derived automatically from voxel_model/D — no separate tunable needed
    nms_dist_frac        = 0.10,
    icp_voxel_frac       = 0.03,
    icp_voxel_min_mm     = 1.0,
    icp_voxel_max_mm     = 10.0,
    max_poses            = 5,
)

def compute_model_diameter(pcd: o3d.geometry.PointCloud) -> float:
    """Approximate diameter as the longest axis of the oriented bounding box."""
    obb  = pcd.get_minimal_oriented_bounding_box()
    return float(np.max(obb.extent))


def estimate_surface_area(pcd: o3d.geometry.PointCloud, D: float) -> float:
    """
    Rough SA estimate from the point cloud when no mesh is available.
    Computes mean nearest-neighbour spacing s, then SA ≈ N × s².
    Falls back gracefully — only used for symmetry / SA:V heuristics
    not for the core PPF parameters.
    """
    pts = np.asarray(pcd.points)
    N   = len(pts)
    if N < 10:
        return np.pi * (D / 2) ** 2   # sphere fallback

    tree   = o3d.geometry.KDTreeFlann(pcd)
    spacings = []
    sample_idx = np.random.choice(N, min(500, N), replace=False)
    for i in sample_idx:
        [_, idx, dists_sq] = tree.search_knn_vector_3d(pts[i], 2)
        if len(dists_sq) > 1:
            spacings.append(np.sqrt(dists_sq[1]))
    s = np.median(spacings) if spacings else D / 50
    return N * s * s




def derive_ppf_params(model_pcd, cfg=None):
    p = {**DEFAULTS, **(cfg or {})}
    D  = compute_model_diameter(model_pcd)
    SA = estimate_surface_area(model_pcd, D)

    voxel_model = float(np.clip(
        np.sqrt(SA / p['n_model_target']),
        D * 0.01, D * 0.10
    ))
    model_down     = model_pcd.voxel_down_sample(voxel_model)
    n_model_actual = len(model_down.points)

    relative_sample_step = voxel_model / D   # used in both train and match

    params = dict(
        D                        = D,
        SA                       = SA,
        voxel_model              = voxel_model,
        n_model_actual           = n_model_actual,
        scene_count_cap          = int(p['n_model_target'] * p['scene_count_cap_k']),
        # constructor
        relative_sample_step     = relative_sample_step,
        relative_dist_step       = p['relative_dist_step'],
        num_angles               = int(p['num_angles']),
        # match()
        relative_scene_sample_step = 1.0 / p['scene_sample_step'],
        relative_scene_distance    = relative_sample_step,  # mirrors training
        # post-processing
        nms_dist                 = D * p['nms_dist_frac'],
        icp_voxel                = float(np.clip(
                                       D * p['icp_voxel_frac'],
                                       p['icp_voxel_min_mm'] * 1e-3,
                                       p['icp_voxel_max_mm'] * 1e-3)),
        max_poses                = p['max_poses'],
    )
    _print_params(params)
    return params


def _print_params(p):
    D_mm = p['D'] * 1e3
    print(f"[PPF auto-params]")
    print(f"  D                    = {D_mm:.1f} mm")
    print(f"  voxel_model          = {p['voxel_model']*1e3:.2f} mm  "
          f"(n_model ≈ {p['n_model_actual']})")
    print(f"  relative_sample_step = {p['relative_sample_step']:.4f} × D  "
          f"= {p['relative_sample_step']*D_mm:.2f} mm")
    print(f"  relative_dist_step   = {p['relative_dist_step']:.3f} × D")
    print(f"  num_angles           = {p['num_angles']}  "
          f"({360/p['num_angles']:.0f}° per bin)")
    print(f"  NMS radius           = {p['nms_dist']*1e3:.2f} mm")
    print(f"  ICP voxel            = {p['icp_voxel']*1e3:.2f} mm")
    print(f"  max_poses            = {p['max_poses']}")