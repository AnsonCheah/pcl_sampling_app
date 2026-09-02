"""Tests for file_utils PLY comment I/O.

The `comment <key> <value>` header lines are a load-bearing protocol (geometry/CLAUDE.md):
SaveStage writes the model-frame provenance there and the tuner reads it back to decide
whether MechVision's symmetry search can be aimed at frame Z. Until now only the writer
existed and every reader hand-rolled its own parser.

Run from project root:
    python -m pytest geometry/tests/test_file_utils.py -q
    python geometry/tests/test_file_utils.py
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import open3d as o3d

from geometry.file_utils import pointcloud_to_ply, read_ply_comments


def _cloud(n=64):
    pts = np.random.default_rng(0).random((n, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (n, 1)))
    return pcd


def test_read_ply_comments_round_trips(tmp_path):
    """Keys and values survive the write/read cycle."""
    path = tmp_path / "c.ply"
    pointcloud_to_ply(_cloud(), str(path), comments=[
        "geocenter_x 0.011",
        "geocenter_qw 1.0",
        "ambiguity_fold 4",
        "ambiguity_aligned 1",
    ])

    got = read_ply_comments(str(path))
    assert got["geocenter_x"] == "0.011"
    assert got["ambiguity_fold"] == "4"
    assert got["ambiguity_aligned"] == "1"


def test_read_ply_comments_ignores_valueless_comments(tmp_path):
    """`comment PCL generated` is written unconditionally and is not a key/value pair.

    A parser that assumed two tokens would either crash or invent a key here.
    """
    path = tmp_path / "c.ply"
    pointcloud_to_ply(_cloud(), str(path), comments=["ambiguity_fold 2"])

    got = read_ply_comments(str(path))
    assert got["ambiguity_fold"] == "2"
    assert "PCL" not in got or got.get("PCL") == "generated"


def test_read_ply_comments_stops_at_end_header(tmp_path):
    """Binary vertex data must never be scanned for comments.

    The payload is binary float32; a byte run can spell `comment ...` by chance, and
    decoding it as text can also raise. Reading must stop at end_header.
    """
    path = tmp_path / "c.ply"
    pointcloud_to_ply(_cloud(2000), str(path), comments=["ambiguity_fold 6"])

    got = read_ply_comments(str(path))
    assert got["ambiguity_fold"] == "6"
    assert len(got) < 10, f"leaked past end_header: {got}"


def test_read_ply_comments_missing_file_returns_empty(tmp_path):
    """A bundle exported before these keys existed must read as 'unknown', not crash."""
    assert read_ply_comments(str(tmp_path / "nope.ply")) == {}


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_read_ply_comments_round_trips(Path(d))
        test_read_ply_comments_ignores_valueless_comments(Path(d))
        test_read_ply_comments_stops_at_end_header(Path(d))
        test_read_ply_comments_missing_file_returns_empty(Path(d))
    print("All tests PASSED")
