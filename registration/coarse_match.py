"""
ppf_matching.py
---------------
PPF + Hough voting using OpenCV ppf_match_3d with geometry-derived parameters.
Drop-in replacement for the FPFH/SHOT + TEASER++ pipeline for the coarse stage.
"""

import cv2
import numpy as np
import open3d as o3d
from registration.ppf_helpers import derive_ppf_params
from scipy.spatial.transform import Rotation as R
import time

CAM_POS = np.array([0,0,0])

def o3d_to_cv_ppf(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Convert Open3D point cloud to the (N, 6) float32 array OpenCV expects:
    [x, y, z, nx, ny, nz] per row. Normals must already be estimated.
    """
    pts = np.asarray(pcd.points,  dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    return np.hstack([pts, nrm])

def extract_pose(cv_pose) -> np.ndarray:
    """
    Extract a valid 4×4 rigid transform from a Pose3DPtr returned by match().

    OpenCV's clusterPoses() has a known unfixed bug where the averaged
    quaternion is not normalized before converting to a rotation matrix.
    The resulting R is non-orthonormal. We fix this by projecting back
    onto SO(3) via SVD: R_fixed = U @ Vt, with det sign correction.
    """
    T = np.array(cv_pose.pose, dtype=np.float64).reshape(4, 4)

    R_raw = T[:3, :3]
    t     = T[:3,  3]

    # Project onto SO(3)
    U, _, Vt = np.linalg.svd(R_raw)
    R_fixed  = U @ Vt

    # Correct for reflection (det = -1 means SVD produced a reflection)
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed   = U @ Vt

    T_fixed          = np.eye(4, dtype=np.float64)
    T_fixed[:3, :3]  = R_fixed
    T_fixed[:3,  3]  = t

    return T_fixed


def poses_to_transforms(poses, max_poses=5):
    transforms = []
    for pose in poses[:max_poses * 4]:
        T      = extract_pose(pose)
        votes  = float(pose.numVotes)
        transforms.append((T, votes))
    return transforms

def prepare_cloud(pcd: o3d.geometry.PointCloud,
                  voxel_size: float,
                  normal_radius: float,
                  max_points: int = None) -> o3d.geometry.PointCloud:
    """Voxel downsample + normal estimation + optional point cap."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    down.orient_normals_consistent_tangent_plane(k=15)
    
    # down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=50)
    # )
    # down.orient_normals_towards_camera_location(CAM_POS)

    if max_points and len(down.points) > max_points:
        idx  = np.random.choice(len(down.points), max_points, replace=False)
        down = down.select_by_index(idx.tolist())

    return down


def run_ppf_matching(
    model_pcd_raw: o3d.geometry.PointCloud,
    scene_pcd_raw: o3d.geometry.PointCloud,
    cfg: dict = None,
) -> list[np.ndarray]:
    """
    Full PPF coarse matching pipeline.

    Returns
    -------
    List of 4×4 numpy arrays (model→scene transforms), ranked by score.
    Empty list if no pose passes the vote threshold.
    """
    start = time.time()
    params = derive_ppf_params(model_pcd_raw, cfg)

    D             = params['D']
    normal_radius = D * 0.05   # generous — normals only need neighbourhood context
    # exit()
    # ── Prepare clouds ────────────────────────────────────────────────────────
    model_down = prepare_cloud(model_pcd_raw,
                               voxel_size=params['voxel_model'],
                               normal_radius=normal_radius)
    print(f"model downsampled has {len(model_down.points)} points")

    # Scene voxel: match model density so PPF features are comparable
    scene_down = prepare_cloud(scene_pcd_raw,
                               voxel_size=params['voxel_model'],
                               normal_radius=normal_radius,
                               max_points=params['scene_count_cap'])
    print(f"scene downsampled has {len(scene_down.points)} points")

    model_cv = o3d_to_cv_ppf(model_down)
    scene_cv = o3d_to_cv_ppf(scene_down)
    print(f"Preprocess took {time.time() - start}s")

    # ── Train PPF model ───────────────────────────────────────────────────────
    print("Instantiating ppf detector")
    print(f"""relativeSamplingStep = {params['relative_sample_step']},
            relativeDistanceStep = {params['relative_dist_step']},
            numAngles            = {params['num_angles']},""")
    detector = cv2.ppf_match_3d_PPF3DDetector(
        relativeSamplingStep = params['relative_sample_step'],   # param 1: resampling step
        relativeDistanceStep = params['relative_dist_step'],     # param 2: distance bin width
        numAngles            = params['num_angles'],             # param 3: angle bin count (integer)
    )
    # detector = cv2.ppf_match_3d_PPF3DDetector(
    #     relativeSamplingStep = params['relative_sample_step'],   # param 1: resampling step
    #     relativeDistanceStep = params['relative_dist_step'],     # param 2: distance bin width
    #     numAngles            = params['num_angles'],             # param 3: angle bin count (integer)
    # )
    print("training model")
    start = time.time()
    detector.trainModel(model_cv)
    print(f"Training took {time.time() - start}s")

    # ── Match ────────────────────────────────────────────────────────────────
    # icp_iterations=0: we handle ICP ourselves with small_gicp / FilterReg
    start = time.time()
    print("performing match")
    print(f"""relativeSceneSampleStep = {params['relative_scene_sample_step']},
            relativeSceneDistance = {params['relative_scene_distance']},
            numAngles            = {params['num_angles']},""")
    poses = detector.match(
        scene_cv,
        relativeSceneSampleStep = params['relative_scene_sample_step'],
        relativeSceneDistance   = params['relative_scene_distance'],
    )
    # poses = detector.match(
    #     scene_cv,
    #     relativeSceneSampleStep = 0.001,
    #     relativeSceneDistance   = 0.001,
    # )

    if not poses:
        return []

    transforms = poses_to_transforms(poses, max_poses=params['max_poses'])

    transforms = _distance_nms(transforms,
                                radius=params['nms_dist'],
                                top_k=params['max_poses'])
    

    for T, votes in transforms:
        R = T[:3, :3]
        col_norms = np.linalg.norm(R, axis=0)          # should be [1, 1, 1]
        det       = np.linalg.det(R)                   # should be +1.0
        ortho_err = np.max(np.abs(R.T @ R - np.eye(3))) # should be < 1e-10
        print(f"votes={votes:.0f}  col_norms={col_norms}  det={det:.6f}  ortho_err={ortho_err:.2e}")

    return [(T, votes) for T, votes in transforms]



    poses = detector.match(scene_cv, relativeSceneSampleStep = 0.2, relativeSceneDistance = 0.03)
    print(f"got {len(poses)} poses")
    print(f"Matching took {time.time() - start}s")

    if not poses:
        print('[PPF] No poses above vote threshold.')
        return []

    # ── Convert to 4×4 and apply NMS ─────────────────────────────────────────
    start = time.time()
    transforms = []
    for pose in poses[:params['max_poses'] * 4]:   # oversample before NMS
        print(pose.t, pose.q)
        print(pose.t, R.from_matrix(pose.pose[:3, :3]).as_quat())        
        transforms.append((pose.pose, float(pose.numVotes)))

    transforms = _distance_nms(transforms,
                                radius=params['nms_dist'],
                                top_k=params['max_poses'])
    print(f"transforms: {transforms}")
    print(f'[PPF] {len(transforms)} poses after NMS '
          f'(from {len(poses)} raw hypotheses)')
    print(f"Post Process NMS took {time.time() - start}s")

    results = [(T, votes) for T, votes in transforms]
    return results


def _distance_nms(
    scored_transforms: list[tuple[np.ndarray, float]],
    radius: float,
    top_k: int,
) -> list[tuple[np.ndarray, float]]:
    """
    Greedy translation-space NMS: keep the highest-voted pose, suppress
    all poses whose translation is within `radius`, repeat.
    Identical in spirit to Mechmind's Distance NMS toggle.
    """
    ranked = sorted(scored_transforms, key=lambda x: x[1], reverse=True)
    print(f"ranked: {ranked}")
    kept   = []
    for T, votes in ranked:
        t = T[:3, 3]
        too_close = any(
            np.linalg.norm(t - k[:3, 3]) < radius
            for k, _ in kept
        )
        if not too_close:
            kept.append((T, votes))
        if len(kept) >= top_k:
            break
    return kept


if __name__=="__main__":
    A_pcd_raw = o3d.io.read_point_cloud('C:/Users/Hmgics/Desktop/pcl_sampling_app/data/972703T000_uniform.ply')
    B_pcd_raw = o3d.io.read_point_cloud('C:/Users/Hmgics/Desktop/pcl_sampling_app/data/97270_real_pcl_00000.ply')
    # o3d.visualization.draw_geometries([A_pcd_raw, B_pcd_raw], width=1080, height=720, zoom=1.0)
    results = run_ppf_matching(A_pcd_raw, B_pcd_raw)
    for result, votes in results:
        print(result, votes)
        B_pcd_raw.transform(np.linalg.inv(result))
        A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0])
        B_pcd_raw.paint_uniform_color([1.0, 1.0, 1.0])
        o3d.visualization.draw_geometries([A_pcd_raw, B_pcd_raw], width=1080, height=720, zoom=1.0)
    # print(results, x)