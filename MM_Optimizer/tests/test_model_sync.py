"""Tests for model_sync.symmetry_metadata -- how the tuner learns whether MechVision's
symmetry search can be aimed at frame Z.

Run from project root:
    python -m pytest MM_Optimizer/tests/test_model_sync.py -q
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import open3d as o3d
import pytest

from MM_Optimizer import model_sync
from geometry.file_utils import pointcloud_to_ply

PART = "pytest_sym_part"


def _write_bundle(root, cloud_type, comments):
    """Write a minimal exported bundle for PART under a fake REF_ROOT."""
    folder = os.path.join(root, PART, f"{PART}_{cloud_type}")
    os.makedirs(folder, exist_ok=True)
    pts = np.random.default_rng(0).random((32, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (32, 1)))
    pointcloud_to_ply(pcd, os.path.join(folder, f"{PART}_{cloud_type}.ply"),
                      comments=comments)


@pytest.fixture
def ref_root(tmp_path, monkeypatch):
    monkeypatch.setattr(model_sync, "REF_ROOT", str(tmp_path))
    return str(tmp_path)


def test_reads_fold_and_aligned_from_the_bundle(ref_root):
    _write_bundle(ref_root, "surface",
                  ["ambiguity_fold 4", "ambiguity_aligned 1"])
    assert model_sync.symmetry_metadata(PART) == (4, True)


def test_unaligned_bundle_still_reports_its_fold(ref_root):
    """The fold is known even when the frame is PCA -- the tuner logs the difference so a
    missed 'Recenter to Ambiguity Axis' is visible rather than silent."""
    _write_bundle(ref_root, "surface",
                  ["ambiguity_fold 4", "ambiguity_aligned 0"])
    assert model_sync.symmetry_metadata(PART) == (4, False)


def test_legacy_bundle_reads_as_c1_unaligned(ref_root):
    """A bundle exported before these keys existed must not enable the symmetry search.

    Defaulting to 'aligned' would point rotationStrategy at a frame nobody verified.
    """
    _write_bundle(ref_root, "surface", ["geocenter_qw 1.0"])
    assert model_sync.symmetry_metadata(PART) == (1, False)


def test_missing_bundle_reads_as_c1_unaligned(ref_root):
    assert model_sync.symmetry_metadata(PART) == (1, False)


def test_continuous_fold_survives_as_zero(ref_root):
    """0 must not be confused with 'missing' -- it selects position-only scoring."""
    _write_bundle(ref_root, "surface",
                  ["ambiguity_fold 0", "ambiguity_aligned 1"])
    assert model_sync.symmetry_metadata(PART) == (0, True)


def test_falls_back_to_the_edge_cloud(ref_root):
    """Both cloud types carry identical values, so either answers for the part."""
    _write_bundle(ref_root, "edge", ["ambiguity_fold 6", "ambiguity_aligned 1"])
    assert model_sync.symmetry_metadata(PART) == (6, True)
