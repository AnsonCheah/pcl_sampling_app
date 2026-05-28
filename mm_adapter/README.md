# mm_adapter

Minimal gRPC client for MechMind Vision Hub. Replaces a large vendored SDK (~104 files) with ~300 lines of stdlib + `grpcio`.

## What it does

- Encodes/decodes MechVision protobuf wire format (varint, length-delimited fields)
- Sets parameters on individual MechVision workflow steps
- Triggers vision runs and returns structured pose results
- Handles unit conversion (mm↔m, degrees↔radians) transparently

## Files

| File | Description |
|------|-------------|
| `mm_adapter.py` | `MechVisionClient` gRPC client |
| `mm_dataclasses.py` | Parameter dataclasses for the 4 MechVision workflow steps |
| `__init__.py` | Re-exports `MechVisionClient` |

## `MechVisionClient`

```python
from mm_adapter import MechVisionClient

client = MechVisionClient(
    hub_address="127.0.0.1:5307",  # local MechVision Hub
    timeout=300,
    z_offset_compatibility_mode=True
)

projects = client.get_projects()        # {name: int_id}
client.set_params(project_id, params)   # push step parameters
result  = client.run_vision(project_id) # trigger run, wait for result
client.close()
```

### `run_vision()` return value

```python
{
    "coarse_poses":      [[x, y, z, qw, qx, qy, qz], ...],
    "coarse_scores":     [float, ...],
    "coarse_time_s":     float,
    "fine_poses":        [[x, y, z, qw, qx, qy, qz], ...],
    "fine_confidences":  [float, ...],
    "fine_time_s":       float,
}
```

Raises `VisionRunError` if MechVision returns `"noCloudInRoi"` or the call times out.

### `set_params()` format

`params` is a nested dict: `{step_name: {param_name: (value_str, type_str, unit_str)}}`.

The adapter handles type coercion:

| `type_str` | Conversion |
|------------|------------|
| `"string"` | passed as-is |
| `"double"` | `float(value_str)` |
| `"bool"` | `value_str == "true"` |

Unit conversion applied before dispatch:

| `unit_str` | Conversion |
|------------|------------|
| `"mm"` | value ÷ 1000 (→ metres) |
| `"rad"` | `math.radians(value)` |
| others | no conversion |

**One parameter per call**: `setStepProperties` calls are issued one at a time. MechVision 1.8.2 is unreliable with batched multi-key updates.

## Workflow steps (`mm_dataclasses.py`)

The four MechVision steps that the optimizer controls:

| Class | MechVision Step | Role |
|-------|-----------------|------|
| `EasyCreateStringList` | `Scene_Path` | Injects the PLY directory path for pre-segmentation |
| `CalcResultsbyPython` | `Pre_Segmentation` | Calls `read_synthetic()` to load and split scene clouds |
| `CoarseMatchingV2` | `Coarse_Match_Synthetics` | 3D coarse registration (PPF + Hough voting) |
| `FineMatchingLite` | `Fine_Match_Synthetics` | Pose refinement (ICP-like) |

Each dataclass field is a 3-tuple `(value, type, unit)`. Call `.to_step_params()` to convert to the dict format expected by `set_params()`.

```python
from mm_adapter.mm_dataclasses import CoarseMatchingV2

coarse = CoarseMatchingV2(
    referencePointSamplingStep=("5", "double", "mm"),
    ...
)
params = {"Coarse_Match_Synthetics": coarse.to_step_params()}
client.set_params(project_id, params)
```

**Typo preserved**: `CoarseMatchingV2.voxelLengthGenetationStrategy` is intentionally misspelled to match the MechVision parameter key exactly.

## Protocol

- Transport: gRPC, method `/mmind.rpc.Json/call`
- Host: `127.0.0.1:5307` (MechVision Hub must be running locally)
- Message: custom `mmind.rpc.Request` (field 1 = JSON action string, field 2 = JSON payload)
- Project IDs are resolved once from `getAppStatus` and cached by name.

## Constraints

- Locked to MechVision 1.8.2 API. The Hub must be running before any client call.
- `z_offset_compatibility_mode=True` is the safe default; setting it `False` changes how vertex Z-offsets are reported by the Hub.
