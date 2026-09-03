# registration/

Two PPF packages. `ppf/` is the vanilla, standalone matcher -- reach for this one.
`ppf_saliency/` is the weighted-voting variant, kept only to re-test the weighting question on
a new part catalogue. Rationale, measurements and usage are in
[README.md](README.md); this file is the list of things not to break.

## Non-Obvious Constraints

**`registration/ppf/` and `registration/_shared/` must import NOTHING local -- not even
`registration.*`.** They are copied out as a **pair**, so both use relative imports
(`from .._shared._backend import ...`). Three tests in `tests/test_ppf.py` enforce it with an
AST walk plus a pin on the third-party surface (numpy / scipy / open3d only). A runtime import
check cannot catch this: the repo root is on `sys.path` for the whole suite, so
`from geometry import ...` would simply work.

**`ppf_saliency/` may import `geometry`, `registration.ppf` and `registration._shared`; it must
not import `sensor`, `physics` or `stages`.** It is not part of the copyable unit.

**`ppf/_geometry.py` and `ppf_saliency/_utils.py` duplicate helpers on purpose.** Duplication
is the price of the package being liftable. If you change a shared version, the one that must
not drift is `model_diameter` -- every part-relative tolerance is normalised against it.

**Parameters are derived, not tuned.** `PPFConfig.derive()` takes a model cloud and a
`SensorProfile` and produces everything. A knob that needs tuning needs tuning for all 5000
parts. Adding a parameter with a hand-picked default is the failure mode to watch for; if a
value cannot be derived from part geometry or a one-time sensor calibration, it belongs in the
small set of explicit application policies (`model_target_points`, `accept_score`).

**The sensor floor is not a second normalisation anchor.** Absolute floors derived from depth
noise are allowed -- they exist for sensor-physics reasons. A second *size* anchor is not.

**Pose metrics are implemented directly, never by importing `bop_toolkit_lib`** -- that would
break the dependency pin. `test_metrics_agree_with_bop_toolkit` checks them element-by-element
where the reference is installed. Three details of the symmetry group are easy to get wrong;
see README.

**Vote deduplication does not subsume the per-bucket cap.** Dedup corrects planar vote *bias*
but runs after the table lookup is expanded, so it does nothing about *cost*. Both mechanisms
are needed. Bucket entries are capped by **striding, never truncation** -- pair enumeration is
ordered by model point, so keeping the first N would retain one contiguous patch and bias every
pose voted from that bin toward it.

**Pose clustering is SE(3), not translation-only.** Translation-only NMS merges two different
orientations of the same part at the same location -- exactly the symmetry-flip case a pose
benchmark has to be able to see.

**Rotations must be re-projected onto SO(3) after averaging** (`_geometry.project_to_so3`). The
mean of several rotation matrices is not one, and an unprojected mean shears the model just
enough to look like a near-miss at verification.

**`_verify` queries a train-time model-frame KD-tree; it does not move the model.** Do not
"simplify" it back to transforming the model. `PPFModel.key_offsets` must likewise survive any
model copy -- `ppf_saliency.with_weights` rebuilds from `__dict__`, so the index rides along.

**The runner-up mask writes into the accumulator in place; do not "fix" it by copying.** `peak`
is read out before the mask is applied and `acc` is reallocated at the top of the next chunk,
so nothing observes the mutation.

**Parallelism is threads, never processes, and it lives in `match_many`.** The hot loop is
NumPy and releases the GIL; threads share the ~40 MB table. Processes measured *worse* than
serial on Windows, which has no `fork`. `match` must stay thread-safe: read the model, allocate
per-call locals only.

**CuPy is optional and must never be imported at module scope.** A test asserts every cupy
import is indented. Availability is *probed* (allocate and reduce on the device), not inferred
from the import succeeding. An absent GPU falls back to NumPy; an unknown backend name raises --
these are different failures, and `MatchResult.backend` reports what actually ran so a benchmark
cannot present CPU timings as GPU ones. Do not make the fallback silent.

**GPU and threads do not stack**; `match_many` defaults to `workers=1` on CuPy. Warm CuPy
kernels before timing anything -- JIT compilation is a process-level one-off.

**Do not build on OpenCV or PCL PPF.** Both have open correctness bugs; see README.

**Weighting lost; that is why it was split out.** Uniform voting beat both weighting schemes
across 2789 instances on 30 T-LESS objects. Do not reintroduce weighted voting into `ppf/`
without evidence from a new catalogue.

**Benchmark harnesses live inside the packages.** `ppf/bench/` so a recipient of the copied
package can reproduce its accuracy claims; `ppf_saliency/bench/` for the weighting ablation.
Repo `bench/` keeps only what cannot move. Both harnesses share
`registration.ppf.bench.dataset` -- there is no second loader, do not add one.
