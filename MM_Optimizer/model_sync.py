"""Copy a part's reference bundle into the MechVision model library.

The sampling app writes `output/reference_pcd/<part>/`; MechVision matches against
`CAD_Match/resource/3d_matching/`. Until now that copy was made by hand, and
`MM_Optimizer/CLAUDE.md` flagged it as a hazard. It was a real one: the deployed copy of
`25333MB000` sat two weeks stale in a superseded model frame (12 056 points, extents
91.0x60.4x51.0) while the app had moved on (26 495 points, 77.8x60.9x68.1). Anything reading
the library got a different part from the one the app produced.

**One library entry per part, holding one cloud.** Coarse and fine always share a cloud type
(see `search_config.REGIMES`), so the regime is expressed by *which* cloud is in the
folder, not by having several:

    3d_matching/<part>/
        <part>.ply             the active regime's cloud, renamed from <part>_<type>.ply
        geo_center.json        identical across cloud types, so nothing is lost by keeping one
        pick_points.json  pick_points_labels.json  poses.poses

The folder is rewritten per regime evaluated, so **regimes must be evaluated sequentially**.
That holds today -- the Phase 1 gate is a sequential loop and the regime is frozen before the
joint Optuna study starts -- but a parallel gate would corrupt the library.

`sync_regime_model` now refuses to deploy a bundle whose model frame disagrees with the scenes
it will be evaluated against (`assert_scene_frames`). The stale copy above was one way to reach
that state; the other is re-exporting a part, since the model frame is only reproducible while
the mesh *and* the sampling settings are unchanged -- a different voxel size or view count can
change which ambiguity axis wins, which moved 25333MB000's frame by 60.6 degrees and 28.3 mm.
Neither failure raises anything on its own: a stale scene still loads and still has a `T_gt`.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import List, Optional, Tuple

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))

REF_ROOT = os.path.join(_ROOT, "output", "reference_pcd")
SYNTH_ROOT = os.path.join(_ROOT, "output", "synthetic_target")
MM_MODEL_ROOT = os.path.join(_DIR, "CAD_Match", "resource", "3d_matching")

# Copied alongside the cloud. Identical across cloud types (SaveStage writes an identity
# geo_center.json and empty pick-point files for every variant), so one copy per part is the
# whole truth rather than a lossy summary.
SIDE_FILES = ("geo_center.json", "pick_points.json", "pick_points_labels.json", "poses.poses")


def source_dir(part: str, cloud_type: str) -> str:
    """Where the app wrote this cloud type for this part."""
    return os.path.join(REF_ROOT, part, f"{part}_{cloud_type}")


def model_dir(part: str) -> str:
    """The part's MechVision library entry."""
    return os.path.join(MM_MODEL_ROOT, part)


def model_ply(part: str) -> str:
    return os.path.join(model_dir(part), f"{part}.ply")


def geo_center(part: str) -> str:
    return os.path.join(model_dir(part), "geo_center.json")


def available_types(part: str, candidates=("surface", "edge")) -> List[str]:
    """Cloud types the app actually exported for this part.

    Asked of the **source** bundle, never the library: the library holds only whichever
    regime was synced last, so asking it what exists would report on history rather than on
    what can be tried.
    """
    return [t for t in candidates
            if os.path.isfile(os.path.join(source_dir(part, t), f"{part}_{t}.ply"))]


def symmetry_metadata(part: str, cloud_type: Optional[str] = None) -> Tuple[int, bool]:
    """`(fold, aligned)` from the exported bundle's PLY header.

    `fold` follows SaveStage's convention: 0 = continuous, 1 = C1 (no rotational symmetry),
    N = N-fold. `aligned` says the cloud was recentred so the dominant ambiguity axis is frame
    Z through the origin -- the precondition for pointing MechVision's `rotationStrategy` at Z.

    Asked of the **source** bundle rather than the library, for the same reason as
    `available_types`: the library holds only whichever regime was synced last.

    A bundle exported before these keys existed carries neither, and reads as `(1, False)` --
    the symmetry search stays off rather than sweeping a line nobody verified. Both cloud types
    carry identical values, so either answers for the part.
    """
    from geometry.file_utils import read_ply_comments

    types = [cloud_type] if cloud_type else available_types(part)
    for t in types:
        ply = os.path.join(source_dir(part, t), f"{part}_{t}.ply")
        comments = read_ply_comments(ply)
        if "ambiguity_fold" in comments:
            return (int(comments["ambiguity_fold"]),
                    comments.get("ambiguity_aligned") == "1")
    return 1, False


class StaleModelFrameError(RuntimeError):
    """A part's scenes and its reference cloud are in different model frames."""


def scene_dirs(part: str, scenes_root: Optional[str] = None) -> List[str]:
    """Every generated scene directory for a part, in name order."""
    root = os.path.join(scenes_root or SYNTH_ROOT, part)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in glob.glob(os.path.join(root, "scene_*")) if os.path.isdir(d))


def check_scene_frames(part: str, model_ply_path: str,
                       scenes_root: Optional[str] = None) -> List[str]:
    """Report which of a part's scenes are in a different model frame from ``model_ply_path``.

    Returns a list of human-readable complaints, empty when everything agrees.

    A scene's ``reference_cloud.ply`` and the bundle's ``<part>_surface.ply`` are the same
    points written in the same run, so on a matching pair this is an exact comparison. When
    they disagree, the scenes' ``T_gt`` values are expressed against a model frame that is
    not the one MechVision will be matching with, and every pose gets scored against the
    wrong reference -- silently, because a stale scene still loads and still has a pose.

    That is not hypothetical. The model frame is only reproducible while the mesh AND the
    settings are unchanged; a different voxel size or view count can change which ambiguity
    axis wins, and on 25333MB000 that moved the frame by 60.6 degrees and 28.3 mm. This
    module's docstring records the same failure reached a different way: a deployed copy that
    sat two weeks stale in a superseded frame.
    """
    import open3d as o3d

    from geometry.geom_utils import reference_frames_agree

    dirs = scene_dirs(part, scenes_root)
    if not dirs or not os.path.isfile(model_ply_path):
        return []
    model = o3d.io.read_point_cloud(model_ply_path)
    problems: List[str] = []
    for d in dirs:
        ref = os.path.join(d, "reference_cloud.ply")
        if not os.path.isfile(ref):
            continue
        why = reference_frames_agree(o3d.io.read_point_cloud(ref), model)
        if why:
            problems.append(f"{d}\n        {why}")
    return problems


def assert_scene_frames(part: str, model_ply_path: str,
                        scenes_root: Optional[str] = None) -> None:
    """``check_scene_frames``, but raise ``StaleModelFrameError`` on any disagreement."""
    problems = check_scene_frames(part, model_ply_path, scenes_root)
    if not problems:
        return
    raise StaleModelFrameError(
        f"{len(problems)} scene(s) for '{part}' are in a different model frame from "
        f"{model_ply_path}.\n    " + "\n    ".join(problems) +
        f"\n    Regenerate this part's scenes against the current bundle "
        f"(bench/generate_scenes.py --only {part} --force), or re-export the bundle from "
        f"the run that produced these scenes.")


def sync_regime_model(part: str, cloud_type: str, check_frames: bool = True) -> str:
    """Install one cloud type as the part's active MechVision model. Returns the PLY path.

    Idempotent, and it *replaces* rather than merges: the destination is cleared first, so
    switching regimes cannot leave the previous cloud behind to be matched against.

    Deploying is the last moment the mismatch is cheap to catch, so the scenes this model
    will be evaluated against are checked first (see ``assert_scene_frames``). Pass
    ``check_frames=False`` only when you know the frames differ and want the copy anyway.
    """
    src = source_dir(part, cloud_type)
    src_ply = os.path.join(src, f"{part}_{cloud_type}.ply")
    if not os.path.isfile(src_ply):
        raise FileNotFoundError(
            f"no {cloud_type} cloud for '{part}' at {src_ply}; "
            f"available: {available_types(part) or 'none'} -- re-run sampling for this part")

    if check_frames:
        # Against the SURFACE bundle, not `src_ply`: scenes write `reference_cloud.ply` from
        # `down_pcd`, which shares its points with `down_pcd_surface`. An edge cloud is a
        # legitimately different point set, so comparing it here would fail every edge regime.
        assert_scene_frames(part, os.path.join(source_dir(part, "surface"),
                                               f"{part}_surface.ply"))

    dst = model_dir(part)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    dst_ply = model_ply(part)
    shutil.copy2(src_ply, dst_ply)
    for name in SIDE_FILES:
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, name))
    return dst_ply
