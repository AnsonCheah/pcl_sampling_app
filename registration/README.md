# registration

Point Pair Feature (PPF) + Hough voting 6D pose matcher, written from scratch.

## Why from scratch

Every off-the-shelf option was unusable for this work:

| Option | Blocker |
|---|---|
| OpenCV `ppf_match_3d` | Heap-corruption bug open since 2015 (opencv_contrib #170); wrong pose in its own official sample (#2034); unnormalised quaternion in `clusterPoses` (#3223). Not installed or declared here either. |
| PCL `PPFRegistration` | Missing `acos` in the feature computation itself (#1171). Maintainers advise against using it. |
| Misc3D (MIT, C++) | Best algorithmic reference available, but the accumulator is a thread-local temporary inside an OpenMP loop — vote diagnostics never reach Python — and there is no way to weight a vote. Needs Open3D *master* circa 2022 and pybind11 2.6; last commit Aug 2022. |
| Open3D | Has no PPF, and never has. |

This work needs accumulator internals (peak margin, supporting model points) and per-point
vote weights, and both live on the far side of a C++ compile step in every alternative.

## Layout

| File | Contents |
|---|---|
| `ppf/config.py` | `SensorProfile`, `PPFConfig.derive()` — parameter derivation |
| `ppf/model.py` | Feature quantisation, sorted-key lookup table, bucket capping |
| `ppf/match.py` | Voting, peak extraction, SE(3) pose clustering, verification |
| `ppf/_frames.py` | Local reference frames, the alpha angle, pose reconstruction |
| `tests/test_ppf.py` | Recovery, symmetry-orbit, weighting and derivation tests |

## Usage

One segmented bin instance at a time:

```python
from registration.ppf import PPFConfig, PPFModel, match, downsample

cfg = PPFConfig.derive(reference_pcd)          # no per-part tuning
m_pts, m_nrm = downsample(model_pts, model_nrm, cfg.tau)
model = PPFModel.train(m_pts, m_nrm, cfg)

result = match(model, cluster_pts, cluster_nrm)
T = result.best.T                              # model frame -> scene frame
```

To weight votes by a per-model-point saliency (e.g. the ambiguity heat map from
`geometry.ambiguity`):

```python
result = match(model.with_weights(w), cluster_pts, cluster_nrm)
```

Uniform weights reproduce the unweighted result exactly — there is a test pinning that, so
any difference measured between weighted and unweighted arms is attributable to the weights
rather than to the code path.

`print(cfg.describe())` shows every derived value and which bound produced it.

## Design commitments

**Zero per-part tunables.** Everything derives from the model cloud plus a one-time
`SensorProfile`. Two *application* policies remain — `model_target_points` (a compute budget)
and `accept_score` — and both are shared across every part. A part whose smallest feature
falls below what the sensor can resolve is reported as `UNDER-RESOLVED` in `cfg.provenance`
rather than silently mis-configured.

**Per-instance, not whole-scene.** Measured on a 26-part bunny bin, whole-scene matching
costs ~50 M scene pair-evaluations extrapolated to 100 parts; per-instance costs ~4 M.

**Diagnostics are first-class.** `Pose.peak_margin` (peak versus runner-up) and `n_support`
are reported per pose because no published PPF work relates accumulator health to model
sparsity — which is exactly what the reference-cloud ablation needs to measure.

## Status

Measured on 26 real segmented instances from `output/synthetic_target/StanfordBunny_fixed/`,
coarse matching only, no ICP refinement:

| Metric | Value |
|---|---|
| Recall @ 5 mm / 10° | 0.77 |
| Recall @ 2 mm / 5° | 0.38 |
| Position error | p50 1.55 mm |
| Rotation error | p50 6.95° |
| Time (NumPy) | 0.24 s / instance |

The tight gate needs a fine-refinement stage, which is not built yet. The NumPy backend is
~6-10x over the 2-4 s per-bin production budget and serves as the correctness oracle for the
planned CuPy backend.

## Tests

```bash
python -m pytest registration/tests -q
```
