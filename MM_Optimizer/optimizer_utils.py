import open3d as o3d
import numpy as np
import os
import time
import re

np.set_printoptions(precision=3, suppress=True)

def read_synthetic(path):
    scene_dir = path[0]
    print(scene_dir)
    pcd_paths = sorted(
        [os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
         if f.startswith('sample_') and f.endswith('.ply')],
        key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
    )
    print(pcd_paths)
    clouds, gt_poses = [], []
    for fpath in pcd_paths:
        pcd = o3d.io.read_point_cloud(fpath)
        pts     = np.array(pcd.points,  dtype=np.float32)
        normals = np.array(pcd.normals, dtype=np.float32)
        clouds.append(np.concatenate([pts, normals, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1))
        gt_poses.append(read_gt_pose_from_ply(fpath))

    print(clouds)
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

if __name__ == "__main__":
    read_synthetic(["C:/Users/Hmgics/Desktop/pcl_sampling_app/output/synthetic_target/25333MB000/scene_00000"])
