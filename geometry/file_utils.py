
from tkinter import Tk, filedialog
from pathlib import Path
import os
import numpy as np
import struct


def list_scene_dirs(part_dir):
    """Sorted `scene_NNNNN` subdirectory names under `part_dir` (a synthetic_target/<part>
    directory). Returns [] if the directory is missing. Base-layer scanner shared by the
    stages that browse on-disk scenes."""
    part_dir = str(part_dir)
    if not os.path.isdir(part_dir):
        return []
    return sorted(
        name for name in os.listdir(part_dir)
        if name.startswith("scene_") and os.path.isdir(os.path.join(part_dir, name)))

def open_source_folder_dialog():
    Tk().withdraw()
    path = filedialog.askdirectory(initialdir=Path.cwd(), title="Select source folder (STL files)")
    return Path(path) if path else None

def pointcloud_to_ply(pcd, ply_path, comments=[]):
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    if len(points) == 0:
        raise ValueError("Empty point cloud")
    if normals.shape[0] != points.shape[0]:
        raise ValueError("Normals missing or size mismatch")

    curvature = np.zeros((points.shape[0], 1), dtype=np.float32)
    vertex_data = np.hstack([points, normals, curvature])

    with open(ply_path, "wb") as f:
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            "comment PCL generated",
            f"element vertex {len(vertex_data)}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float curvature",
            "element face 0",
            "element camera 1",
            "property float view_px",
            "property float view_py",
            "property float view_pz",
            "property float x_axisx",
            "property float x_axisy",
            "property float x_axisz",
            "property float y_axisx",
            "property float y_axisy",
            "property float y_axisz",
            "property float z_axisx",
            "property float z_axisy",
            "property float z_axisz",
            "property float focal",
            "property float scalex",
            "property float scaley",
            "property float centerx",
            "property float centery",
            "property int viewportx",
            "property int viewporty",
            "property float k1",
            "property float k2"
        ]
        for line in header_lines:
            f.write((line + "\n").encode("ascii"))

        for comment in comments:
            f.write(("comment " + str(comment) + "\n").encode("ascii"))
        f.write(("end_header" + "\n").encode("ascii"))

        # --- Vertex block ---
        for row in vertex_data:
            f.write(struct.pack("<7f", *row))

        # --- Camera block ---
        camera_floats = [
            0.0, 0.0, 1.0,     # view point
            1.0, 0.0, 0.0,     # x axis
            0.0, 1.0, 0.0,     # y axis
            0.0, 0.0, 1.0,     # z axis
            525.0,            # focal
            1.0, 1.0,         # scale
            320.0, 240.0      # center
        ]

        for v in camera_floats:
            f.write(struct.pack("<f", v))

        f.write(struct.pack("<i", 640))  # viewportx
        f.write(struct.pack("<i", 480))  # viewporty
        f.write(struct.pack("<f", 0.0))  # k1
        f.write(struct.pack("<f", 0.0))  # k2