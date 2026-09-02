import open3d as o3d
import numpy as np
import os
import shutil
import time
import re

np.set_printoptions(precision=3, suppress=True)


def list_synthetic_scenes(part_dir):
    """Return one group of PLY paths per scene_MMMMM directory (M-scene convention).

    Each inner list contains all sample_*.ply files for one bin capture
    (one M-scene).  These are fed as a list of N individual clouds to
    MechVision's read_synthetic, which returns N point clouds -- one per
    instance in the bin.

    Parameters
    ----------
    part_dir : str
        Part-level directory containing scene_MMMMM sub-directories, or a
        single scene directory if no sub-directories exist.

    Returns
    -------
    scenes : List[List[str]]
        One inner list per scene_MMMMM directory, ordered by scene index.
        Each inner list is sorted by sample (instance) index.
    """
    subdirs = sorted(
        d for d in os.listdir(part_dir)
        if re.match(r'^scene_\d+$', d) and os.path.isdir(os.path.join(part_dir, d))
    )
    if not subdirs:
        # Already a single scene directory -- wrap as one M-scene
        plys = sorted(
            [os.path.join(part_dir, f) for f in os.listdir(part_dir)
             if f.startswith('sample_') and f.endswith('.ply')],
            key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
        )
        return [plys] if plys else []

    groups = []
    for subdir in subdirs:
        scene_path = os.path.join(part_dir, subdir)
        plys = sorted(
            [os.path.join(scene_path, f) for f in os.listdir(scene_path)
             if f.startswith('sample_') and f.endswith('.ply')],
            key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
        )
        if plys:
            groups.append(plys)
    return groups


def read_single_ply(path):
    """Load a single PLY file for the Pre_Segmentation step.

    Called by MechVision's Calc Results by Python (Pre_Segmentation).
    Mirrors read_synthetic's return format so the same output port works.

    Parameters
    ----------
    path : list[str]
        path[0] = absolute path to the scene .ply file to load.

    Returns
    -------
    clouds : list of np.ndarray, shape [(N, 7)]
        Single-element list containing [x, y, z, nx, ny, nz, 0] array.
        List wrapper matches read_synthetic's port type.
    """
    ply_file = path[0]
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
    pts     = np.array(pcd.points,  dtype=np.float32)
    normals = np.array(pcd.normals, dtype=np.float32)
    cloud   = np.concatenate(
        [pts, normals, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1
    )
    return [cloud]  # list of one, matching read_synthetic port type


def read_synthetic(path):
    scene_dir = path[0]
    pcd_paths = sorted(
        [os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
         if f.startswith('sample_') and f.endswith('.ply')],
        key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
    )
    clouds, gt_poses = [], []
    for fpath in pcd_paths:
        pcd = o3d.io.read_point_cloud(fpath)
        pts     = np.array(pcd.points,  dtype=np.float32)
        normals = np.array(pcd.normals, dtype=np.float32)
        clouds.append(np.concatenate([pts, normals, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1))
        gt_poses.append(read_gt_pose_from_ply(fpath))

    return clouds

def reshape_coarse_pose_list(coarse_pose_list, ground_truth, tik):
    return coarse_pose_list[0]


def read_gt_pose_from_ply(ply_path, scalar_first=True):
    gt = {}
    with open(ply_path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if line == "end_header":
                break
            # Match comment gt_* entries
            match = re.match(r"comment\s+(gt_\w+)\s+([-+eE0-9\.]+)", line)
            if match:
                key, value = match.groups()
                gt[key] = float(value)

    required_keys = ['gt_x', 'gt_y', 'gt_z', 'gt_qx', 'gt_qy', 'gt_qz', 'gt_w']
    for k in required_keys:
        if k not in gt:
            raise ValueError(f"Missing {k} in PLY header")

    if scalar_first:
        return np.asarray([gt['gt_x'], gt['gt_y'], gt['gt_z'], gt['gt_w'], gt['gt_qx'], gt['gt_qy'], gt['gt_qz']]).astype(np.double)
    else:
        return np.asarray([gt['gt_x'], gt['gt_y'], gt['gt_z'], gt['gt_qx'], gt['gt_qy'], gt['gt_qz'], gt['gt_w']]).astype(np.double)

def pre_coarse():
    stamp = time.time()
    # print(f"[pre_coarse] {stamp}")
    return [stamp]

def post_coarse(tik):
    stamp = time.time()
    # print(f"[post_coarse] {stamp}")
    return [stamp - tik[0]]

def pre_fine():
    stamp = time.time()
    # print(f"[pre_fine] {stamp}")
    return [stamp]


def post_fine(tik):
    stamp = time.time()
    # print(f"[post_fine] {stamp}")
    return [stamp - tik[0]]
