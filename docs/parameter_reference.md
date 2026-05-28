# MechMind Parameter Reference — Layer 1 Curated RAG

> **Maintenance rule**: this file is read by the local LLM but NEVER written by it.
> Updates are proposed by a frontier LLM reviewing experience-bank data and approved
> by a human before merge. Local 14B models read this; frontier LLMs (Claude/GPT-4) write it.
>
> RAG version tracked in experience_bank records as `rag_version` (git commit hash).

---

## Three-way name mapping

| MechMind UI label | Python adapter key | OpenCV PPF equivalent |
|---|---|---|
| Distance Quantification | `distQuantification` | `relativeDistanceStep` |
| (model sampling step) | `refStep` | `relativeSamplingStep` |
| Angle Quantification | `angleQuantification` | `numAngles` (≈ 2π / angleStep) |
| Scene sampling | `referredStep` | `relativeSceneSampleStep` |
| Max Vote Ratio | `maxVoteRatio` | Hough cluster threshold |
| Use Distance NMS | `useDistanceNMS` | relativeSceneDistance + cluster filter |
| Voxel verification | `minVoxelLength`, `maxVoxelLength`, `outputNum` | (proprietary) |
| Axis pose filter | `filterCandidatePoseByAxis`, `angleThreshold` | (proprietary) |
| Fine op approach | `operationApproach` | (proprietary) |
| Deviation correction | `deviationCorrectionCapacity` | (proprietary) |

---

## Coarse Matching Parameters

### refStep / Model Sampling Step

**Python adapter**: `refStep`
**MechMind UI**: "Reference Step" (model sampling density)
**OpenCV PPF equivalent**: `relativeSamplingStep` (PPF3DDetector constructor)
**source_ref**: https://docs.opencv.org/4.x/d9/d25/classcv_1_1ppf__match__3d_1_1PPF3DDetector.html

**PPF mechanism**: Controls the fraction of model points used to build the Hough
voting table. Lower `refStep` = finer model sampling = denser feature table =
higher discriminative power but slower indexing and larger memory footprint.
OpenCV source comment: "Step value for discretizing the model volume / reference to total extent."

**Valid range**: integer 1–20 (MechMind hard limit). `refStep=1` is finest (all model points).
`refStep=10` uses every 10th point on the model surface.

**Practical guidance**:
- Small parts (< 50 mm): start at 4–6 for enough points on the Hough table
- Medium parts (50–200 mm): 6–12 typical sweet spot
- Large parts (> 200 mm): 10–18, speed is critical

**Interacts with**: `referredStep` (must satisfy `referredStep ≤ refStep`),
`distQuantification` (both control Hough space resolution jointly),
`maxNumOfPointPairsPerFeature` (pairs drawn from model samples)

---

### distQuantification / Distance Quantification

**Python adapter**: `distQuantification`
**MechMind UI**: "Distance Quantification"
**OpenCV PPF equivalent**: `relativeDistanceStep` (PPF3DDetector constructor)
**source_ref**: https://docs.opencv.org/4.x/d9/d25/classcv_1_1ppf__match__3d_1_1PPF3DDetector.html

**PPF mechanism**: Hough-space bin size for the distance feature. The PPF descriptor
is a 4-tuple (f1..f4) where f1 is the distance between two surface points. This
parameter sets the bin width for f1 in the Hough accumulator.
OpenCV source: "Relative step size in Hough space for distance [0.025..0.05 typical]."

**Valid range**: [0.5, 3.0] — MechMind scales this as a ratio of the normalised
model diameter. Value 1.0 is the MechMind optimal default.

**Practical guidance**:
- Lower (0.5–0.8): finer discrimination, slower, better for complex geometry with
  many distinctive surface features
- Optimal (1.0): MechMind calibrated default, good starting point
- Higher (1.5–3.0): coarser, faster, tolerates sensor noise on featureless surfaces
  but risks missing weak Hough peaks on low-texture parts

**Interacts with**: `angleQuantification` (both define Hough space resolution),
`refStep` (coarser refStep → sparser voting, pair with coarser distQ to avoid
sparse table + fine bins)

---

### angleQuantification / Angle Quantification

**Python adapter**: `angleQuantification`
**MechMind UI**: "Angle Quantification"
**OpenCV PPF equivalent**: `numAngles` (inversely: more angles = finer quantisation)
**source_ref**: https://docs.opencv.org/4.x/d9/d25/classcv_1_1ppf__match__3d_1_1PPF3DDetector.html

**PPF mechanism**: Number of angle bins in the Hough accumulator for the angle
features (f2..f4). More bins = finer angular resolution = slower but more
discriminative. OpenCV default: 30 (= 6° bins). MechMind uses a different
parameterisation — the value is passed as the bin count directly.

**Valid range**: {60, 90, 120, 180} (MechMind allowed discrete values)
Finer = 180 (2° bins), coarser = 60 (6° bins).

**Practical guidance**:
- 60: fast, good for noisy scenes or large parts
- 90: good default for most industrial parts
- 120–180: for high-precision requirements or highly symmetric parts

---

### referredStep / Scene Sampling Step

**Python adapter**: `referredStep`
**MechMind UI**: "Referred Step" (scene sampling density)
**OpenCV PPF equivalent**: `relativeSceneSampleStep`

**PPF mechanism**: Fraction of scene points used as query points for Hough voting.
Lower = more scene points queried = slower but higher recall.

**HARD CONSTRAINT**: `referredStep ≤ refStep` always. Violating this causes the
MechVision run to be rejected. The adapter will raise an error.

**Valid range**: integer 1–20.

**Practical guidance**:
- Typically set to `refStep / 2` as a starting point
- For cluttered scenes (many false positives): lower referredStep to increase recall
- For clean scenes (single part): can equal refStep (minimal scene sampling)

---

### maxNumOfPointPairsPerFeature / Point Pair Count

**Python adapter**: `maxNumOfPointPairsPerFeature`
**MechMind UI**: "Max Number of Point Pairs Per Feature"
**OpenCV PPF equivalent**: (controls voting density)

**PPF mechanism**: Maximum number of oriented point pairs sampled per model feature
point. More pairs = denser Hough voting = higher recall but slower.

**Valid range**: typically 1000–20000. Warm-start: 5000 (small parts), 10000 (large parts).
Optimizer uses multipliers {0.25, 0.5, 1.0, 2.0, 4.0} × warm-start value.

---

### maxVoteRatio / Max Vote Ratio

**Python adapter**: `maxVoteRatio`
**MechMind UI**: "Max Vote Ratio"
**OpenCV PPF equivalent**: threshold on the relative Hough peak height

**PPF mechanism**: Minimum fraction of the maximum vote count that a pose hypothesis
must receive to be retained as a candidate. Higher = stricter = fewer but more
confident pose candidates. Lower = more permissive = higher recall at cost of
processing the extra candidates downstream.

**Valid range**: [0.5, 0.9]

**Practical guidance**:
- 0.5–0.6: high recall, good for occluded scenes
- 0.7: balanced default
- 0.8–0.9: high precision, use only when false positives are a problem

---

### useDistanceNMS / Distance NMS

**Python adapter**: `useDistanceNMS`
**MechMind UI**: "Use Distance NMS"
**OpenCV PPF equivalent**: non-maximum suppression on Hough peak clustering

**PPF mechanism**: Whether to suppress nearby pose candidates that vote for the
same peak. Disabling NMS returns more candidates (higher recall but more overlap).

**Valid range**: {True, False}. Default: True.

---

### voxelLengthRange / Voxel Verification

**Python adapter**: `minVoxelLength`, `maxVoxelLength` (paired)
**MechMind UI**: "Voxel Length Range"
**OpenCV PPF equivalent**: (proprietary — no PPF analog)

**Mechanism**: After Hough peak selection, each candidate pose is verified by
voxel-grid overlap between the model and the scene. These parameters control
the voxel grid resolution. Smaller voxels = finer verification = slower.

**Valid range**: geometry-derived. Warm-start: 0.5–2% of model diameter for min,
2–5% for max. Optimizer uses multipliers {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}.

---

### outputNum / Output Count

**Python adapter**: `outputNum`
**MechMind UI**: "Output Num"
**OpenCV PPF equivalent**: (proprietary)

**Mechanism**: Number of pose candidates returned by coarse matching. Set to the
expected number of instances in the scene. Higher values increase cycle time
proportionally.

**Valid range**: integer 1–3 in the optimiser search space.

---

### filterCandidatePoseByAxis / Axis Filter (Edge mode only)

**Python adapter**: `filterCandidatePoseByAxis`
**MechMind UI**: "Filter Candidate Pose By Axis"
**OpenCV PPF equivalent**: (proprietary)

**Mechanism**: In edge-mode coarse matching, optionally restrict candidate poses
to those whose principal axis aligns with a gravity vector or reference direction.
Only meaningful for edge-mode pipelines.

**Valid range**: {True, False}. Only tested when `coarse_mode = 1.0` (Edge).

---

## Fine Matching Parameters

### operationApproach / Operation Approach

**Python adapter**: `operationApproach`
**MechMind UI**: "Operation Approach"
**OpenCV PPF equivalent**: (proprietary fine-matching variant)

**Mechanism**: Controls the ICP/fine-matching strategy. Higher values trade speed
for accuracy.

| Value | Label | Use case |
|---|---|---|
| 0 | HighSpeed | Very fast, lower accuracy |
| 1 | Standard | Balanced default |
| 2 | HighAccuracy | Slower, better for precise requirements |
| 3 | ExtraHighAccuracy | Slowest, best for tight tolerances |

**Phase 3 selection heuristic** (from `search_config.py`):
- coarse position error < 3mm → try [HighSpeed, Standard]
- 3–10mm → try [Standard, HighAccuracy]
- > 10mm → try [HighAccuracy, ExtraHighAccuracy]

---

### deviationCorrectionCapacity / Deviation Correction

**Python adapter**: `deviationCorrectionCapacity`
**MechMind UI**: "Deviation Correction Capacity"
**OpenCV PPF equivalent**: (controls ICP convergence radius)

**Mechanism**: How much translational/rotational deviation the fine-matcher
corrects from the coarse pose estimate. Higher = larger correction basin
(can handle poorer coarse poses) but slower convergence.

| Value | Label |
|---|---|
| 0 | Low |
| 1 | Medium |
| 2 | High |

---

### scoreLevel / Score Level

**Python adapter**: `scoreLevel`
**MechMind UI**: "Score Level"

**Mechanism**: Controls the stringency of the fine-matching quality score.
Used together with `confidenceThreshold` to filter low-quality poses.
Optimiser sweeps this with `confidenceThreshold=0` first to find the best level,
then optimises `confidenceThreshold`.

**Valid range**: {0.0, 1.0, 2.0, 3.0}

---

### confidenceThreshold / Confidence Threshold

**Python adapter**: `confidenceThreshold`
**MechMind UI**: "Confidence Threshold"

**Mechanism**: Minimum confidence score for a fine-matched pose to be accepted.
Poses below this threshold are discarded. Higher = stricter, fewer false positives,
lower recall.

**Valid range**: [0.0, 0.6]. Optimised after `scoreLevel` is locked.

---

### onlyConsiderVisibleSurfaceOfModel

**Python adapter**: `onlyConsiderVisibleSurfaceOfModel`
**Valid range**: {True, False}

**Mechanism**: When True, only the part of the model visible from the camera
(above the sensor horizon) is used for fine matching. Improves accuracy for
heavily occluded parts in deep bins.

---

### considerErrorofNormalAngles

**Python adapter**: `considerErrorofNormalAngles`
**Valid range**: {True, False}

**Mechanism**: Whether to include normal-angle error in the fine-matching objective.
Useful for parts where surface orientation is a strong distinguishing feature.

---

## Symmetry-Aware Parameter Guidance

When `symmetry_class` is known from `mesh_analysis.analyze_mesh()`:

| symmetry_class | Angular param guidance |
|---|---|
| `ASYMMETRIC` | All angular params are meaningful — optimise normally |
| `C2` | 180° ambiguity — enable rotation search in Phase 4 |
| `C3` | 120° ambiguity — Phase 4 with 3-fold angle step |
| `C4` | 90° ambiguity — Phase 4 with 4-fold angle step |
| `C6` | 60° ambiguity — Phase 4 with 6-fold angle step |
| `SO2` | Continuous axis rotation is meaningless — do NOT optimise angular params for rotation about symmetry axis; use VSD coverage metric |
| `SO3` | All rotations equivalent — position accuracy only; skip Phase 3–4 angular params entirely |

---

## Critical Interaction Rules

1. **`referredStep ≤ refStep`** — hard constraint, always enforced. Never violate.
2. **distQ and angleQ are coupled** — both control Hough space resolution. Finer
   distQ with coarser angleQ is allowed but may produce uneven discrimination.
3. **maxVoteRatio after voting params** — tune maxVoteRatio only after refStep,
   distQ, and angleQ are stable. Earlier tuning conflates Hough quality with threshold.
4. **outputNum is most downstream** — tune last; it scales cycle time linearly.
5. **scoreLevel before confidenceThreshold** — always lock scoreLevel first (with
   confThresh=0), then tune confidenceThreshold. Reverse order wastes trials.

---

*Bootstrap note*: This file was seeded from OpenCV PPF documentation and MechMind
adapter code. Fields marked `(proprietary)` have no public OpenCV analog.
The `PPF mechanism` text is sourced from OpenCV 4.x documentation and source
comments (`ppf_match_3d/src/`), not paraphrased from training data.
Human review required before any update is merged.
