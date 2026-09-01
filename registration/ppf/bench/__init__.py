"""Benchmark harness for the vanilla PPF matcher: scene loading, pose metrics, the sweep.

Lives inside the package rather than beside it because a matcher whose accuracy claims are
not reproducible by whoever received the code is not really shippable.  Copy ``ppf/`` into
another project and ``python -m ppf.bench.run --all`` still works.

It reads a *directory format* -- ``<root>/<part>/scene_*/`` holding ``reference_cloud.ply``,
``sample_<i>.ply`` and ``sample_<i>.npz`` -- not a particular project's output tree.  Point
it anywhere with ``--scenes-root`` or ``PPF_SCENES_ROOT``.

What is deliberately *not* here: scene generation.  Producing those directories needs a
physics engine, a sensor simulator and a segmentation stage, which is a far heavier
dependency set than the matcher itself; it stays in the surrounding repository
(``bench/generate_scenes.py``).  This half only consumes the format.
"""

from .dataset import Instance, Scene, list_parts, list_scenes, load_reference, load_scene
from .metrics import (LOOSE, TIGHT, PoseError, add, adi, evaluate_pose, mssd, summarise,
                      symmetry_transforms_from_bop)

__all__ = [
    "Instance", "Scene", "list_parts", "list_scenes", "load_reference", "load_scene",
    "PoseError", "evaluate_pose", "summarise", "symmetry_transforms_from_bop",
    "mssd", "add", "adi", "TIGHT", "LOOSE",
]
