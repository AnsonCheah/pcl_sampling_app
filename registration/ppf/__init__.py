"""Vanilla Point Pair Feature matching -- a standalone, dependency-free implementation.

Standalone means exactly that: this package imports **nothing** from the surrounding
repository.  Its only third-party dependencies are NumPy, SciPy and Open3D, so the directory
can be copied into another project and used as-is.  The three shared geometry helpers it
needs are inlined in ``_geometry.py``.

Written rather than wrapped because every available implementation is unusable for this
work: OpenCV's ``ppf_match_3d`` has an open heap-corruption bug from 2015, a wrong-pose bug
in its own official sample, and an unnormalised quaternion in ``clusterPoses``; PCL's
``PPFRegistration`` is missing an ``acos`` in the feature itself and its maintainers say not
to use it; and Misc3D -- the one maintained C++ option -- destroys the accumulator inside an
OpenMP loop, so vote-level diagnostics never reach Python.

Typical use, one segmented instance at a time::

    from registration.ppf import PPFConfig, PPFModel, match, downsample

    cfg = PPFConfig.derive(model_pcd)                 # no per-part tuning
    m_pts, m_nrm = downsample(model_pts, model_nrm, cfg.tau)
    model = PPFModel.train(m_pts, m_nrm, cfg)
    result = match(model, cluster_pts, cluster_nrm)
    T = result.best.T                                 # model frame -> scene frame

What "vanilla" excludes
    Weighted voting, per-model-point saliency, and the view-dependent ambiguity heat map.
    Measured across 2789 instances on the 30 T-LESS objects, no weighting scheme beat plain
    uniform voting: PPF-descriptor saliency tied on the loose gate and lost on the tight one
    (0.29 vs 0.35) for 32% more time, and the ambiguity heat map lost outright (0.56 vs
    0.66).  Pruning the reference cloud was worse again -- curvature-only reached 0.16.  That
    machinery now lives in ``registration.ppf_saliency``, which is kept only so the question
    can be re-tested on a new part catalogue.

What "vanilla" still includes
    Drost's formulation plus the two Hinterstoisser (ECCV 2016) corrections that are about
    the matcher's own cost and bias rather than about weighting -- vote deduplication and
    steep-pair re-admission -- along with feature-bin spreading and the per-bucket cap.
    Each is an individually ablatable toggle on ``PPFConfig``.  They are not optional in
    practice: without the cap, a 100x30x20 box expands to ~1.8e9 votes for a single instance.

Parameters are derived, never tuned
    ``PPFConfig.derive`` takes a model cloud and a ``SensorProfile`` and produces everything,
    recording in ``.provenance`` which bound actually bound each value.  There are no
    per-part knobs -- that is the point, because a knob that needs tuning needs tuning for
    all 5000 parts.
"""

from .._shared._backend import cupy_available
from .._shared._frames import alpha_of, frames_to_x, pose_from_correspondence
from .config import MECHVISION_NAMES, PPFConfig, SensorProfile
from .match import MatchResult, Pose, downsample, match, match_many
from .model import PPFModel, pair_features

__all__ = [
    "PPFConfig",
    "SensorProfile",
    "MECHVISION_NAMES",
    "PPFModel",
    "pair_features",
    "match",
    "match_many",
    "downsample",
    "cupy_available",
    "Pose",
    "MatchResult",
    "frames_to_x",
    "alpha_of",
    "pose_from_correspondence",
]
