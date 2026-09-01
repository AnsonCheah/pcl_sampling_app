"""Convex decomposition worker isolated for out-of-process execution.

``vhacdx.compute_vhacd`` (the backend of ``trimesh.decomposition.convex_decomposition``)
holds the GIL for its entire multi-second runtime. Running it in the calling thread therefore
starves the Open3D GUI main thread, so queued progress-bar callbacks never render until the
call returns. We run it in a separate process instead (see DecomposeStage.worker); the parent
thread blocks on the future, which releases the GIL and keeps the GUI responsive.

This module deliberately imports only numpy + vhacdx (no Open3D / trimesh / GUI) so the spawned
child process stays cheap to start.
"""
import numpy as np
from vhacdx import compute_vhacd


def vhacd_decompose(vertices, faces, **kwargs):
    """Run VHACD on a triangle mesh in a worker process.

    Parameters
    ----------
    vertices : (V, 3) float array
    faces    : (F, 3) int array of triangle vertex indices
    **kwargs : VHACD parameters forwarded to ``vhacdx.compute_vhacd``

    Returns a list of ``(vertices, faces)`` tuples, one per convex hull. Plain numpy
    arrays are returned (not Open3D geometry) so the result pickles back to the parent.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    # VHACD wants vtkPolyData-style faces: each row prefixed with its vertex count (3).
    vtk_faces = (
        np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces))
        .ravel()
        .astype(np.uint32)
    )
    return [(v, f) for v, f in compute_vhacd(vertices, vtk_faces, **kwargs)]
