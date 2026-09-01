"""Ablation harness for the weighted-voting variant: reference-cloud and vote-weight arms.

Lives beside the package it measures rather than in the repo's ``bench/``, which is reserved
for things that are not PPF testing.  Unlike ``registration.ppf.bench`` this half is **not**
standalone and is not meant to be: the arms it compares are defined by the ambiguity heat map
and by curvature, so it imports ``geometry.ambiguity`` directly.  The cloud-analysis helpers
the arms need are inlined in ``registration/ppf_saliency/_utils.py``.

The scene *loader* is shared with the vanilla package (``registration.ppf.bench.dataset``)
rather than duplicated -- both halves read the same directory format, and two copies would
drift.  ``SYNTH_ROOT`` here anchors that loader to this repository's output tree, which the
package-side default deliberately cannot assume.

Run::

    python -m registration.ppf_saliency.bench.ablation --all --out ablation.json
    python -m registration.ppf_saliency.bench.visualize_ppf
"""

from __future__ import annotations

import os

# registration/ppf_saliency/bench/__init__.py -> up four levels is the repository root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", ".."))
SYNTH_ROOT = os.path.join(REPO_ROOT, "output", "synthetic_target")
MESH_ROOT = os.path.join(REPO_ROOT, "mesh_raw")

__all__ = ["REPO_ROOT", "SYNTH_ROOT", "MESH_ROOT"]
