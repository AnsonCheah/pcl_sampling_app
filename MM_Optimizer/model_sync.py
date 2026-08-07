"""Copy a part's reference bundle into the MechVision model library.

The sampling app writes `output/reference_pcd/<part>/`; MechVision matches against
`CAD_Match/resource/3d_matching/`. Until now that copy was made by hand, and
`MM_Optimizer/CLAUDE.md` flagged it as a hazard. It was a real one: the deployed copy of
`25333MB000` sat two weeks stale in a superseded model frame (12 056 points, extents
91.0x60.4x51.0) while the app had moved on (26 495 points, 77.8x60.9x68.1). Anything reading
the library got a different part from the one the app produced.

**One library entry per part, holding one cloud.** Coarse and fine always share a cloud type
(see `search_config.PHASE1_REGIMES`), so the regime is expressed by *which* cloud is in the
folder, not by having several:

    3d_matching/<part>/
        <part>.ply             the active regime's cloud, renamed from <part>_<type>.ply
        geo_center.json        identical across cloud types, so nothing is lost by keeping one
        pick_points.json  pick_points_labels.json  poses.poses

The folder is rewritten per regime evaluated, so **regimes must be evaluated sequentially**.
That holds today — the Phase 1 gate is a sequential loop and the regime is frozen before the
joint Optuna study starts — but a parallel gate would corrupt the library.
"""

from __future__ import annotations

import os
import shutil
from typing import List

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))

REF_ROOT = os.path.join(_ROOT, "output", "reference_pcd")
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


def sync_regime_model(part: str, cloud_type: str) -> str:
    """Install one cloud type as the part's active MechVision model. Returns the PLY path.

    Idempotent, and it *replaces* rather than merges: the destination is cleared first, so
    switching regimes cannot leave the previous cloud behind to be matched against.
    """
    src = source_dir(part, cloud_type)
    src_ply = os.path.join(src, f"{part}_{cloud_type}.ply")
    if not os.path.isfile(src_ply):
        raise FileNotFoundError(
            f"no {cloud_type} cloud for '{part}' at {src_ply}; "
            f"available: {available_types(part) or 'none'} — re-run sampling for this part")

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
