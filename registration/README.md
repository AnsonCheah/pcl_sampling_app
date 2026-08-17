# registration

Point Pair Feature (PPF) + Hough voting 6D pose matcher, written from scratch.

Two packages, deliberately separate:

| Package | What it is |
|---|---|
| **`ppf/`** | **Vanilla PPF. Standalone — imports nothing from this repo.** Use this one. |
| `ppf_saliency/` | The same matcher plus weighted voting and saliency. Kept to re-test the weighting question on a new catalogue; it does not currently win. |

## Why from scratch

Every off-the-shelf option was unusable for this work:

| Option | Blocker |
|---|---|
| OpenCV `ppf_match_3d` | Heap-corruption bug open since 2015 (opencv_contrib #170); wrong pose in its own official sample (#2034); unnormalised quaternion in `clusterPoses` (#3223). Not installed or declared here either. |
| PCL `PPFRegistration` | Missing `acos` in the feature computation itself (#1171). Maintainers advise against using it. |
| Misc3D (MIT, C++) | Best algorithmic reference available, but the accumulator is a thread-local temporary inside an OpenMP loop — vote diagnostics never reach Python. Needs Open3D *master* circa 2022 and pybind11 2.6; last commit Aug 2022. |
| Open3D | Has no PPF, and never has. |

This work needs accumulator internals (peak margin, supporting model points), and those live
on the far side of a C++ compile step in every alternative.

## `ppf/` — the standalone package

Its contract is that it depends on **NumPy, SciPy and Open3D and nothing else**, so the
directory can be copied into another project unchanged. Two tests enforce that rather than
trusting it: `test_package_imports_nothing_from_this_repository` walks the AST of every
source file (a runtime check cannot catch this — the repo is always on `sys.path` during the
suite), and `test_third_party_dependencies_are_only_numpy_scipy_open3d` pins the dependency
surface so a future convenience import cannot quietly widen it.

| File | Contents |
|---|---|
| `ppf/config.py` | `SensorProfile`, `PPFConfig.derive()` — parameter derivation |
| `ppf/model.py` | Feature quantisation, sorted-key lookup table, bucket capping |
| `ppf/match.py` | Voting, peak extraction, SE(3) pose clustering, verification |
| `ppf/_frames.py` | Local reference frames, the alpha angle, pose reconstruction |
| `ppf/_geometry.py` | Inlined `model_diameter` / `median_spacing` / `project_to_so3` |
| `ppf/bench/` | Scene loader, symmetry-aware pose metrics, the benchmark sweep |
| `tests/test_ppf.py` | Standalone-ness, recovery, symmetry orbit, derivation, metrics |

`ppf/_geometry.py` duplicates three helpers from `geometry/geom_utils.py` on purpose. That is
the price of the package being liftable; the one that must not drift is `model_diameter`,
because every part-relative tolerance is normalised against it.

### Usage

One segmented bin instance at a time:

```python
from registration.ppf import PPFConfig, PPFModel, match, downsample

cfg = PPFConfig.derive(reference_pcd)          # no per-part tuning
m_pts, m_nrm = downsample(model_pts, model_nrm, cfg.tau)
model = PPFModel.train(m_pts, m_nrm, cfg)

result = match(model, cluster_pts, cluster_nrm)
T = result.best.T                              # model frame -> scene frame
```

`print(cfg.describe())` shows every derived value and which bound produced it.

### Benchmarking

The harness ships **inside** the package, because a matcher whose accuracy claims cannot be
reproduced by whoever received the code is not really shippable:

```bash
python -m registration.ppf.bench.run --all \
    --scenes-root output/synthetic_target \
    --models-info mesh_raw/tless/models_info.json
```

It reads a directory format (`<root>/<part>/scene_*/`), not this repo's output tree — point
it anywhere with `--scenes-root` or `PPF_SCENES_ROOT`. Scene *generation* is the other half
and stays in the repo (`bench/generate_scenes.py`), because producing those directories needs
MuJoCo, the sensor simulator and the segmentation stage — a far heavier dependency set than
the matcher itself.

MSSD / ADD / ADI and the BOP symmetry group are implemented directly rather than by importing
`bop_toolkit_lib`, to hold the dependency surface. `test_metrics_agree_with_bop_toolkit`
checks them element-by-element against the reference wherever it happens to be installed.

## Design commitments

**Zero per-part tunables.** Everything derives from the model cloud plus a one-time
`SensorProfile`. Two *application* policies remain — `model_target_points` (a compute budget)
and `accept_score` — and both are shared across every part. A part whose smallest feature
falls below what the sensor can resolve is reported as `UNDER-RESOLVED` in `cfg.provenance`
rather than silently mis-configured.

**Per-instance, not whole-scene.** Measured on a 26-part bunny bin, whole-scene matching
costs ~50 M scene pair-evaluations extrapolated to 100 parts; per-instance costs ~4 M.

**Vanilla keeps Drost + Hinterstoisser, not textbook Drost alone.** Vote deduplication,
feature-bin spreading, steep-pair re-admission and the per-bucket cap are all still here as
individually ablatable toggles on `PPFConfig`. They are about the matcher's own cost and bias,
not about weighting: without the cap a 100x30x20 box expands to ~1.8e9 votes for one instance.

**Diagnostics are first-class.** `Pose.peak_margin` (peak versus runner-up) and `n_support`
are reported per pose because no published PPF work relates accumulator health to model
sparsity.

## Measured performance

Vanilla `ppf/`, coarse matching only, no ICP refinement. 9 T-LESS parts x 2 synthetic bin
scenes plus the in-house `25333MB000`, **1702 instances**, scored with
`python -m registration.ppf.bench.run --all`:

| part | class | n | BOP (MSSD<0.2D) | @5mm/10° | @2mm/5° | MSSD p50 | s/inst |
|---|---|---|---|---|---|---|---|
| obj_000019 | discrete | 108 | 0.954 ±0.040 | 0.907 | 0.565 | 3.66 mm | 0.157 |
| obj_000017 | continuous | 31 | 0.935 ±0.086 | 0.935 | 0.774 | 2.60 mm | 0.215 |
| 25333MB000 | unknown | 541 | 0.906 ±0.025 | 0.867 | 0.553 | 3.58 mm | 0.148 |
| obj_000021 | asymmetric | 97 | 0.845 ±0.072 | 0.825 | 0.660 | 3.01 mm | 0.195 |
| obj_000022 | asymmetric | 97 | 0.845 ±0.072 | 0.835 | 0.629 | 3.02 mm | 0.194 |
| obj_000018 | asymmetric | 38 | 0.789 ±0.130 | 0.579 | 0.263 | 8.13 mm | 0.247 |
| obj_000027 | discrete | 29 | 0.724 ±0.163 | 0.552 | 0.241 | 10.01 mm | 0.300 |
| obj_000005 | discrete | 85 | 0.471 ±0.106 | 0.471 | 0.376 | 78.52 mm | 0.165 |
| obj_000013 | continuous | 352 | 0.389 ±0.051 | 0.276 | 0.190 | 24.33 mm | 0.166 |
| obj_000001 | continuous | 324 | 0.148 ±0.039 | 0.108 | 0.086 | 25.52 mm | 0.210 |
| **ALL** | | **1702** | **0.624** | 0.568 | 0.384 | 15.93 mm | 0.176 |
| | asymmetric | 232 | 0.836 | 0.789 | 0.582 | 3.86 mm | 0.203 |
| | discrete | 222 | 0.739 | 0.694 | 0.450 | 33.15 mm | 0.179 |
| | continuous | 707 | 0.303 | 0.228 | 0.168 | 23.92 mm | 0.189 |

`s/inst` above is single-threaded. With `--workers 8` the same sweep runs at **0.050
s/instance** with byte-identical accuracy; see below.

This reproduces the earlier 30-part ablation's shape (uniform arm 0.66 overall, continuous
the weak class) on an independently generated scene set.

**"Continuous symmetry" is not itself the cause of the weak class**, and reading the class
average that way would send tuning in the wrong direction. `obj_000017` is continuous and
reaches 0.935; the failures are `obj_000001` (0.148) and `obj_000013` (0.389), which are the
two *smallest* parts (63.5 mm and 58.1 mm) and yield ~1800 scene points per instance against
`obj_000017`'s ~9300.

Occlusion is **not** the driver either — checked rather than assumed. Per-instance visibility
is comparable across the whole set (overlap p50 0.21-0.34), and `obj_000019` has the *lowest*
visibility of any part (0.207) together with the *highest* recall (0.954).

`obj_000005` fails differently and is worth separating: MSSD p50 of 78.5 mm against a 108.7 mm
diameter is most of a part length, so those are systematic flips into a pose BOP's single
discrete symmetry does not cover — not near-misses. A tighter verification stage would attack
that; more points would not.

The `@2mm/5°` column needs a fine-refinement stage, which is not built yet. At 0.050
s/instance threaded, a 50-instance bin now takes ~2.5 s, which is at the edge of the 2-4 s
per-bin production budget rather than far outside it.

### Parallelism and the GPU

`match_many(model, clusters, workers=N)` matches a bin's instances concurrently, and
`--workers` exposes it on the benchmark runner. Combined with the fixes below, end-to-end
matching went **0.389 -> 0.050 s/instance (7.7x)** over the same 1702 instances, with every
accuracy figure unchanged.

**Threads, not processes.** The hot loop is NumPy — sort, bincount, searchsorted, gathers —
all of which release the GIL, and threads share the ~40 MB trained table instead of copying
it. Processes measured *worse than serial* on Windows, where the absence of `fork` makes
every worker respawn the interpreter and retrain:

| mode | s/inst | speedup |
|---|---|---|
| serial | 0.147 | 1.00x |
| **threads x8** | **0.042** | **3.47x** |
| threads x16 | 0.042 | 3.50x |
| processes x8 | 0.120 | 1.23x |

Scaling plateaus at 8 threads; the ceiling is the Python remainder (the chunk loop, the
greedy clustering), not memory bandwidth.

**The CuPy backend works but does not stack with threads.** `backend="cupy"` runs the vote
stage on the GPU (pose reconstruction, clustering and verification stay on the host, since
they use scipy). Measured on an RTX 5090, per instance:

| part | serial numpy | serial cupy | threads x8 numpy | threads x4 cupy |
|---|---|---|---|---|
| obj_000019 | 0.140 | **0.051** | **0.041** | 0.048 |
| obj_000001 | 0.195 | **0.054** | 0.055 | 0.057 |
| obj_000021 | 0.195 | **0.058** | 0.055 | 0.085 |

So the GPU is worth roughly what 8 CPU threads are worth, and combining them gains nothing —
one host thread already saturates the device, so `match_many` defaults to `workers=1` on the
CuPy backend. Use the GPU for single-instance *latency*; use threads for bin *throughput*.

CuPy is an **optional** dependency, never imported at module scope (a test enforces that), so
the package still imports and runs on a machine with no CUDA build. Asking for a GPU that is
not there falls back to NumPy rather than failing — but never silently: `MatchResult.backend`
records what actually ran, so a benchmark cannot report CPU timings as GPU ones. An unknown
backend name still raises.

*Measurement note:* CuPy JIT-compiles kernels on first use. An early version of this
comparison put that one-off cost inside the first timed block and made the GPU look 5x slower
than the CPU on one part and 4x faster on another. Warm the kernels before timing anything.

### Cost structure

`PPFConfig.derive` (0.02-0.19 s) and `PPFModel.train` (0.9-1.7 s) run **once per part**, and
scene loading (PLY + normal estimation, ~0.2 s) once per scene; only `match` is per instance.
Across 351 instances that is 6.2 s of fixed cost against 111.6 s of matching — training per
instance instead would add ~405 s.

Three train-time/vectorisation fixes took matching from **0.389 to 0.176 s/instance (2.21x)**
with **every accuracy figure bit-identical** (verified by diffing the two result JSONs across
all 10 parts, and by comparing poses against the pre-change implementations):

| fix | share of match time before |
|---|---|
| `lookup` via a CSR key index instead of two `searchsorted` passes | 42% |
| `_cluster` vectorised over representatives; SO(3) projection batched | ~20% |
| runner-up mask applied in place instead of copying the accumulator | 8% |
| `_verify` queries a train-time model-frame KD-tree instead of rebuilding one per pose | ~9% |

The `_verify` fix uses the same reasoning as the key index: a rigid transform preserves
distances, so mapping the scene into the model frame lets every candidate pose share one tree
built once at train time, instead of rebuilding a tree over the moved model ~16 times per
instance.

## Why weighting was stripped out

Measured across 2789 instances on the 30 T-LESS objects (`bench/ablation.py`):

| arm | BOP recall (MSSD<0.2D) | @2mm/5deg | s/inst |
|---|---|---|---|
| **uniform** | **0.66** | **0.35** | 0.182 |
| PPF-descriptor saliency | 0.64 | 0.29 | 0.240 |
| ambiguity heat map | 0.56 | 0.20 | 0.240 |
| curvature + heat pruning | 0.40 | 0.09 | 0.102 |

No weighting or pruning scheme beat plain uniform voting. That machinery therefore lives in
`ppf_saliency/`, not in the matcher anyone should reach for first.

## Tests

```bash
python -m pytest registration/tests -q
```

## Why the saliency knee excludes the zero-score atom

The ambiguity heat map is not a smooth distribution. A large share of points are explained
*exactly* by some ambiguity transform and score a hard 0.0 — measured at 41.5% on T-LESS
`obj_000018`. That spike is a vertical jump in the CDF at its left edge, and therefore by far
the furthest point from the chord, so `find_cdf_knee` lands *on* the atom, the threshold comes
back as 0.0, and `v >= 0` keeps every point.

The pruning arm then silently becomes a copy of the baseline, and the ablation reports that
pruning is harmless — because pruning never happened. Those zero-scoring points are exactly
what the arm means to drop, so `arms.py` removes them first and fits the knee to the rest.
