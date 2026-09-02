"""Point Pair Feature matching, built from scratch.

Written rather than wrapped because every available implementation is unusable for this
work: OpenCV's ``ppf_match_3d`` has an open heap-corruption bug from 2015, a wrong-pose bug
in its own official sample, and an unnormalised quaternion in ``clusterPoses``; PCL's
``PPFRegistration`` is missing an ``acos`` in the feature itself and its maintainers say not
to use it; and Misc3D -- the one maintained C++ option -- destroys the accumulator inside an
OpenMP loop, so the vote-level diagnostics this project needs never reach Python, and it
offers no way to weight a vote.

Typical use, one segmented bin instance at a time::

    from registration.ppf_saliency import PPFConfig, PPFModel, match, downsample

    cfg = PPFConfig.derive(model_pcd)                 # no per-part tuning
    m_pts, m_nrm = downsample(model_pts, model_nrm, cfg.tau)
    model = PPFModel.train(m_pts, m_nrm, cfg)
    result = match(model, cluster_pts, cluster_nrm)
    T = result.best.T                                 # model frame -> scene frame

Weighted voting is supported via ``model.with_weights(w)`` and reproduces the unweighted
result exactly at uniform weights.

**Use uniform unless you have evidence for your own parts.** Measured across 2789 instances
on the 30 T-LESS objects, no weighting scheme beat plain uniform voting: PPF-descriptor
saliency tied on the loose gate and lost on the tight one (0.29 vs 0.35) for 32% more time,
and the ambiguity heat map lost outright (0.56 vs 0.66). Pruning the reference cloud was far
worse again -- curvature-only reached 0.16. The weighting machinery is kept because it is the
only way to test the question on a new part catalogue, not because it currently wins.
"""

from .._shared._backend import cupy_available
from .._shared._frames import alpha_of, frames_to_x, pose_from_correspondence
from .config import MECHVISION_NAMES, PPFConfig, SensorProfile
from .match import MatchResult, Pose, downsample, match, match_many
from .model import PPFModel, pair_features
from .saliency import combine, ppf_saliency, transfer_weights

__all__ = [
    "match_many",
    "cupy_available",
    "PPFConfig",
    "SensorProfile",
    "MECHVISION_NAMES",
    "PPFModel",
    "pair_features",
    "match",
    "downsample",
    "Pose",
    "MatchResult",
    "frames_to_x",
    "alpha_of",
    "pose_from_correspondence",
    "ppf_saliency",
    "transfer_weights",
    "combine",
]
