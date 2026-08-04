# registration/

Two PPF packages. `ppf/` is the vanilla, **standalone** matcher — reach for this one.
`ppf_saliency/` is the older weighted-voting variant, kept only to re-test the weighting
question on a new part catalogue.

## Non-Obvious Constraints

**`registration/ppf/` must import NOTHING local — not even `registration.*`.** This is
stricter than the old "may import `geometry`" rule, and it is the whole point of the package:
it has to survive being copied into another project. Two tests enforce it
(`registration/tests/test_ppf.py`): an AST walk over every source file, and a pin on the
third-party surface (numpy / scipy / open3d only). A *runtime* import check cannot catch this
— the repo root is on `sys.path` for the entire suite, so `from geometry import ...` would
simply work. Inside the package, use **relative** imports (`from .. import PPFConfig`);
`registration.ppf.bench.dataset` would resolve here and nowhere else.

**`ppf/_geometry.py` duplicates three helpers from `geometry/geom_utils.py`, on purpose.**
`model_diameter`, `median_spacing`, `project_to_so3`. Duplication is the price of the package
being liftable. If you change the shared version, the one that must not drift is
`model_diameter` — every part-relative tolerance is normalised against it, and two disagreeing
diameter conventions were already in use once (longest minimal-OBB extent vs AABB diagonal:
94.4 mm vs 146.5 mm on the bunny), so "5% of diameter" meant two different things depending on
which module you stood in.

**`ppf_saliency/` may import `geometry`; it must still not import `sensor`, `physics` or
`stages`.** It consumes `AmbiguityProfile` directly and needs the shared geometry helpers.

**Benchmark harnesses live inside the packages; the repo's `bench/` holds only what cannot
move.** `registration/ppf/bench/` reads a directory format (`<root>/<part>/scene_*/`) and holds
the pose metrics and the sweep, so a recipient of the copied package can reproduce its accuracy
claims. `registration/ppf_saliency/bench/` holds the weighting ablation (`ablation.py`,
`arms.py`, `visualize_ppf.py`, its own ambiguity-aware `metrics.py`). Repo `bench/` keeps only
`generate_scenes.py`, `validate_ambiguity.py` and `fetch_dataset.py` — producing scene
directories needs MuJoCo, the sensor simulator and the segmentation stage — and is reserved for
an Optuna tuning benchmark (deferred). **There is no `bench/dataset.py` any more**; both
harnesses share `registration.ppf.bench.dataset`, and `registration/ppf_saliency/bench/__init__.py`
supplies the repo-anchored `SYNTH_ROOT` that the package-side default cannot assume. Do not
re-add a second loader.

**`ppf_saliency/_utils.py` inlines `find_cdf_knee`, `mean_curvature` and `extract_edge_points`**
so the ablation moves as one unit with the package it measures. Unlike `ppf/_geometry.py` this
is not a standalone requirement — `ppf_saliency` may import `geometry`, and `ablation.py` and
`visualize_ppf.py` still import `geometry.ambiguity` because the arms are *defined* by the heat
map. `find_cdf_knee` here deliberately does not print, unlike the original.

**Pose metrics are implemented directly, never by importing `bop_toolkit_lib`** — that would
break the dependency pin. `test_metrics_agree_with_bop_toolkit` checks them element-by-element
against the reference where it is installed. Three details of BOP's
`get_symmetry_transformations` were each got wrong once and each silently changes the group:
`max_sym_disc_step` is **a fraction of the diameter, not an angle** (step count is
`ceil(pi/step)`, not `ceil(2*pi/step)`); the discretised continuous set **includes the identity**
(`i` from 0); and when a continuous axis exists the output is **only** the composed transforms —
appending the bare discrete ones duplicates every one of them. Also: `adi` runs
ground-truth → estimated, and nearest-neighbour is asymmetric (29.4 mm vs 25.2 mm on the same
pose if reversed).

**Do not build on OpenCV or PCL PPF.** OpenCV `ppf_match_3d` has an open heap-corruption bug
from 2015 (opencv_contrib #170), a wrong-pose bug in its own official sample (#2034), and an
unnormalised quaternion in `clusterPoses` (#3223). PCL's `PPFRegistration` is missing an
`acos` in the feature itself (#1171) and its maintainers advise against using it. Misc3D is
worth reading but destroys its accumulator inside an OpenMP loop, so vote diagnostics never
reach Python.

**Parameters are derived, not tuned.** `PPFConfig.derive()` takes a model cloud and a
`SensorProfile` and produces everything. There are no per-part tunables — that is the point,
because a knob that needs tuning needs tuning for all 5000 parts. Adding a parameter with a
hand-picked default is the failure mode to watch for; if a value cannot be derived from part
geometry or a one-time sensor calibration, it belongs in the small set of explicit
*application policies* (`model_target_points`, `accept_score`) and must be justified there.

**The sensor floor is not a second normalisation anchor.** Absolute floors derived from depth
noise are allowed, for the same reason `AmbiguityConfig` allows them — *"the absolute floors
exist for sensor-physics reasons, not as size rules."* What is forbidden is a second *size*
anchor.

**`tau` is bisected on the real downsampled point count, never from a surface-area formula.**
`tau = sqrt(SA/M)` with `SA ~ N*s^2` is wrong by ~4.5x on randomly sampled clouds (a Poisson
process has median NN distance `0.4697/sqrt(density)`, not `1/sqrt(density)`). Model point
count goes as `tau^-2` and work as its square, so a 2x error in `tau` is a ~20x error in
runtime. This bit the module it replaced.

**Vote deduplication does not subsume the per-bucket cap.** Dedup runs *after* the table
lookup is expanded, so it corrects planar vote *bias* but not *cost*. A 100x30x20 box has six
distinct normals; its feature bins reach 12 000 entries and one instance expands to ~1.8e9
votes without `PPFConfig.max_bucket_entries`. Both mechanisms are needed and they solve
different problems. Neither was stripped by "vanilla" — that only removed weighting.

**Bucket entries are capped by striding, never truncation.** Pair enumeration is ordered by
model point, so keeping the first N entries would retain one contiguous patch of the part and
bias every pose voted from that bin toward it.

**Pose clustering is SE(3), not translation-only.** Translation-only NMS merges two different
orientations of the same part at the same location — exactly the symmetry-flip case a pose
benchmark has to be able to see.

**Rotations must be re-projected onto SO(3) after averaging** (`ppf/_geometry.project_to_so3`).
The mean of several rotation matrices is not one, and an unprojected mean scales and shears the
model just enough to look like a near-miss at verification.

**`PPFModel.lookup` indexes, it does not search — and `key_offsets` must survive any model
copy.** The key space is a dense integer range (`quantise` packs four clipped bin indices, so
every key is `< n_dist_bins * (n_angle+1)**3`), so bin boundaries are tabulated once at train
time as CSR offsets. Two `searchsorted` passes over the ~2 M-entry key array were **42% of
match time**; the table is 19-35x faster for ~5-7 MB per part. It falls back to binary search
above `MAX_KEY_INDEX` (8 M elements) because the key space grows as `n_angle^3` — at
`n_angle=180` it would want 616 MB to index 2 M pairs. Both paths return identical ranges, and
tests assert that on *empty* bins too, which are most of the key space and where an off-by-one
would return a neighbour's entries. `ppf_saliency.with_weights` rebuilds from `__dict__`, so
the index rides along automatically — a test pins that, since dropping it silently costs only
speed and would not otherwise be noticed.

**The runner-up mask writes into the accumulator in place; do not "fix" it by copying.** The
copy was 8% of match time. `peak` is read out before the mask is applied and `acc` is
reallocated at the top of the next chunk, so nothing observes the mutation.

**`_verify` queries a train-time model-frame KD-tree; it does not move the model.** A rigid
transform preserves distances, so scoring `dist(scene, R*model + t)` is identical to
`dist(R^T(scene - t), model)` — and the second form queries `PPFModel.point_tree`, built once,
instead of rebuilding a tree per candidate pose (~16 per instance). Do not "simplify" it back
to transforming the model.

**Parallelism is threads, never processes, and it lives in `match_many`.** The hot loop is
NumPy (sort, bincount, searchsorted, gathers), all of which release the GIL, and threads share
the ~40 MB table. Measured 3.47x at 8 threads; **processes measured 1.23x — worse than the
thread pool and barely better than serial** — because Windows has no `fork`, so every worker
respawns the interpreter and retrains the model. Scaling plateaus at 8 (the Python remainder:
chunk loop, greedy clustering). `match` is thread-safe because it only reads the model and
allocates per-call locals, including the accumulator it masks in place — keep it that way.

**CuPy is optional and must never be imported at module scope.** A top-level `import cupy`
turns an accelerator into a hard requirement and breaks the package on any machine without a
CUDA build; a test asserts every cupy import is indented. Availability is *probed* (allocate
and reduce on the device), not inferred from the import succeeding — a half-written wheel here
imported fine with four `.pyd` files missing and reported a bogus "circular import".

**An absent GPU falls back to NumPy; an unknown backend name raises.** These are different
failures. The fallback is safe only because `MatchResult.backend` reports what actually ran —
that is what replaced the old blanket `NotImplementedError`, and it is what stops a benchmark
presenting CPU timings as GPU ones. Do not make the fallback silent.

**GPU and threads do not stack; `match_many` defaults to `workers=1` on CuPy.** One host
thread already saturates the device, and several pushing the same stream serialise while
adding synchronisation. Serial CuPy ~= 8 CPU threads (~0.05 s/instance either way). GPU for
single-instance latency, threads for bin throughput.

**Warm CuPy kernels before timing anything.** JIT compilation is a process-level one-off. Put
it inside the first timed block and the GPU appears 5x *slower* than the CPU on one part and
4x faster on another — that exact mistake was made and corrected here.

**Dedup uses sort-then-diff, not `np.unique`.** NumPy 2.x routes `unique` through a hash table
that is pathologically slow on wide int64 keys: 0.411 s vs 0.020 s on 1.4 M values, and it is
the hottest call in the matcher.

**Weighting lost; that is why it was split out.** Across 2789 instances on 30 T-LESS objects,
uniform voting reached 0.66 BOP recall against 0.64 for PPF-descriptor saliency and 0.56 for
the ambiguity heat map, and pruning arms reached 0.16-0.40. Do not reintroduce weighted voting
into `ppf/` without evidence from a new catalogue.
