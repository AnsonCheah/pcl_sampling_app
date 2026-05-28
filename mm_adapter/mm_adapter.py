"""
Minimal MechMind Vision adapter — gRPC client to MechMind Hub.

Only two operations: set step parameters and trigger a vision run.
Replaces the entire mm_src/ vendored SDK (104 files, PySide2/SNAP7/protobuf deps)
with ~300 lines using only grpcio + stdlib.

Dependencies: grpcio
Python: 3.6+
"""
import json
import logging
import math
import time
import grpc
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUB_ADDRESS = "127.0.0.1:5307"
GRPC_METHOD = "/mmind.rpc.Json/call"
MECH_VISION = "Mech-Vision"
DEFAULT_TIMEOUT = 300  # seconds

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protobuf wire-format helpers (mmind.rpc.Request / Reply)
#
# Request: field 1 = string msg (tag 0x0a), field 2 = bytes data (tag 0x12)
# Reply:   field 1 = bytes  msg (tag 0x0a)
# ---------------------------------------------------------------------------

def _encode_varint(value):
    """Encode an unsigned integer as a protobuf varint."""
    buf = b""
    while value > 0x7F:
        buf += bytes([(value & 0x7F) | 0x80])
        value >>= 7
    buf += bytes([value & 0x7F])
    return buf


def _decode_varint(data, pos):
    """Decode a varint from *data* starting at *pos*. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, pos


def _length_delimited(field_number, payload):
    """Encode one length-delimited protobuf field."""
    tag = (field_number << 3) | 2  # wire type 2
    return bytes([tag]) + _encode_varint(len(payload)) + payload


def serialize_request(msg_str, data=None):
    """Serialize mmind.rpc.Request(msg, data) to raw protobuf bytes."""
    buf = _length_delimited(1, msg_str.encode("utf-8"))
    if data:
        buf += _length_delimited(2, data)
    return buf


def deserialize_reply(raw):
    """Deserialize mmind.rpc.Reply bytes → the msg field (bytes)."""
    if not raw:
        return b""
    pos = 0
    while pos < len(raw):
        tag_byte = raw[pos]
        field_number = tag_byte >> 3
        wire_type = tag_byte & 0x07
        pos += 1
        if wire_type == 2:  # length-delimited
            length, pos = _decode_varint(raw, pos)
            field_data = raw[pos:pos + length]
            pos += length
            if field_number == 1:
                return field_data
        else:
            # Skip unknown wire types (varint=0, 64-bit=1, 32-bit=5)
            if wire_type == 0:
                _, pos = _decode_varint(raw, pos)
            elif wire_type == 1:
                pos += 8
            elif wire_type == 5:
                pos += 4
    return b""


# ---------------------------------------------------------------------------
# Unit / type conversion  (mirrors original convert_type behaviour)
# ---------------------------------------------------------------------------


def convert_type(value, type_str="string", unit_str=""):
    """Cast *value* (always a string from params_dict) and apply unit conversion.

    type_str: "string" | "double" | "bool"
    unit_str: "" | "m" | "mm" | "rad" | "°"  (degree symbol = no conversion)
    """
    if type_str == "bool":
        converted = str(value).strip().lower() in ("true", "1", "yes")
    elif type_str == "double":
        converted = float(value)
    else:
        converted = str(value)

    if isinstance(converted, (int, float)):
        if unit_str == "m":
            converted = converted / 1000.0
        elif unit_str == "rad":
            converted = math.radians(converted)

    return converted


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MechVisionError(Exception):
    """Base exception for MechVision communication errors."""


class ProjectNotFoundError(MechVisionError):
    """Raised when project_id does not map to a loaded project."""


class VisionRunError(MechVisionError):
    """Raised when a vision run fails (no point cloud, no result, timeout)."""


# ---------------------------------------------------------------------------
# MechVisionClient
# ---------------------------------------------------------------------------


class MechVisionClient(object):
    """Minimal gRPC client for MechMind Hub → MechVision interaction.

    Usage::

        client = MechVisionClient()
        client.set_params(2, {
            "Coarse_Match_Synthetics": {
                "maxScenePointNum": ("1000000", "double", ""),
                ...
            }
        })
        result = client.run_vision(2)
        client.close()
    """

    def __init__(self, hub_address=HUB_ADDRESS, timeout=DEFAULT_TIMEOUT,
                 z_offset_compatibility_mode=True):
        self.timeout = timeout
        self.z_offset_compatibility_mode = z_offset_compatibility_mode
        self._channel = grpc.insecure_channel(hub_address)
        self._call = self._channel.unary_unary(
            GRPC_METHOD,
            request_serializer=lambda x: x,    # already bytes
            response_deserializer=lambda x: x,  # return raw bytes
        )
        self._projects = {}  # cache: {project_name: int_id}

    # -- low-level gRPC ----------------------------------------------------

    def _call_hub(self, function, payload=None, timeout=None):
        """Send a function call to Hub, return parsed JSON dict (or raw bytes)."""
        msg = {"function": function}
        if payload:
            msg.update(payload)
        raw_request = serialize_request(json.dumps(msg))
        raw_reply = self._call(raw_request, timeout=timeout or self.timeout)
        reply_bytes = deserialize_reply(raw_reply)
        return reply_bytes

    def _call_service(self, service_name, function, payload=None, timeout=None):
        """Route a call to a named service through Hub's 'forward' mechanism."""
        inner_msg = {"function": function}
        if payload:
            inner_msg.update(payload)
        return self._call_hub(
            "forward",
            {"name": service_name, "message": inner_msg},
            timeout=timeout,
        )

    def _call_vision(self, function, payload=None, project_name=None, timeout=None):
        """Call a function on a MechVision project service."""
        if payload is None:
            payload = {}
        if function == "run":
            payload["z_offset_compatibility_mode"] = self.z_offset_compatibility_mode
            payload.setdefault("source_of_initial_jps", 1)
        return self._call_service(project_name, function, payload, timeout)

    # -- public API --------------------------------------------------------

    def get_projects(self, timeout=None):
        """Return ``{project_name: int_id}`` of loaded MechVision projects."""
        reply_bytes = self._call_service(MECH_VISION, "getAppStatus", timeout=timeout)
        result = json.loads(reply_bytes.decode("utf-8"))
        projects_raw = result.get("projects_id", {})
        self._projects = {v: int(k) for k, v in projects_raw.items()}
        return self._projects

    def get_project_name(self, project_id):
        """Resolve *project_id* (int) to its project name. Refreshes cache if miss."""
        # Search for project_id in the values of the inverted dict
        for project_name, pid in self._projects.items():
            if pid == project_id:
                return project_name

        # Refresh cache and try again
        self.get_projects()
        for project_name, pid in self._projects.items():
            if pid == project_id:
                return project_name

        raise ProjectNotFoundError(f"Project ID {project_id} not found. Loaded: {self._projects}")

    def set_params(self, project_id, params_dict):
        """Set step properties on a MechVision project.

        *params_dict*: ``{step_name: {param_key: (value, type_str, unit_str)}}``

        Each parameter is sent as an individual ``setStepProperties`` gRPC call
        (MechVision does not reliably support batched multi-key updates).
        """
        project_name = self.get_project_name(project_id)
        for step_name, params in params_dict.items():
            for param_key, param_tuple in params.items():
                converted = convert_type(param_tuple[0], param_tuple[1], param_tuple[2])
                msg = {"name": step_name, "values": {param_key: converted}}
                reply = self._call_vision("setStepProperties", msg, project_name=project_name)
                log.debug(f"setStepProperties {step_name}.{param_key} = {converted}  →  {reply}")


    def run_vision(self, project_id, timeout=None):
        t_start = time.time()
        project_name = self.get_project_name(project_id)
        reply_bytes = self._call_vision("run", {}, project_name, timeout)
        t_wall = time.time() - t_start
        result = json.loads(reply_bytes.decode("utf-8"))
        
        if result.get("noCloudInRoi"):
            raise VisionRunError("No point cloud in ROI")

        # Empty poses = valid "no detection" result — let caller decide how to score it.
        coarse_time_raw = result.get("coarse_time_s", [t_wall])
        fine_time_raw   = result.get("fine_time_s",   [0.0])

        return {
            "coarse_poses":      result.get("coarse_poses",      []),
            "coarse_scores":     result.get("coarse_scores",     []),
            "coarse_time_s":     coarse_time_raw[0] if coarse_time_raw else t_wall,
            "fine_poses":        result.get("fine_poses",        []),
            "fine_confidences":  result.get("fine_confidences",  []),
            "fine_time_s":       fine_time_raw[0]   if fine_time_raw   else 0.0,
        }

    def close(self):
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from mm_dataclasses import *
    from pathlib import Path
    from rich import print as rp
    
    logging.basicConfig(level=logging.INFO)

    client = MechVisionClient()
    start = time.time()
    projects = client.get_projects()
    logging.info(f"Projects retrieved successfully in {time.time() - start}")

    REF_NAME = "25333MB000"
    PROJ_NAME = "CAD_Match"
    WORKING_DIR = Path.cwd()
    MM_REF_PATH = f"{WORKING_DIR}/MM_Optimizer/{PROJ_NAME}/resource/3d_matching/{REF_NAME}"
    SYN_SCENE_PATH = f"{WORKING_DIR}/output/synthetic_target"

    project_id = client._projects[PROJ_NAME]

    start = time.time()
    # Both coarse and fine load the same model file and geocenter file.
    # Edge matching model is not compatible with surface matching model file.
    model_selection    = (f"{REF_NAME}_surface", "string", "")
    model_file_name    = (f"{MM_REF_PATH}_surface/{REF_NAME}_surface.ply", "string", "")
    geo_center_file    = (f"{MM_REF_PATH}_surface/geo_center.json", "string", "")

    scene  = EasyCreateStringList(strings=(f"{SYN_SCENE_PATH}/{REF_NAME}/scene_00000/", "string", ""))
    pre_seg_py = CalcResultsbyPython(
        name="Pre_Segmentation",
        scriptFilePath=(f"C:/Users/Hmgics/Desktop/pcl_sampling_app/MM_Optimizer/optimizer_utils.py", "string", ""),
        funcName=(f"read_synthetic", "string", ""),
        )
    coarse = CoarseMatchingV2(
        name = "Coarse_Match_Synthetics",
        modelSelection=model_selection,
        modelFileName=model_file_name,
        geoCenterFileName=geo_center_file,
    )
    fine = FineMatchingLite(
        name = "Fine_Match_Synthetics",
        modelSelection=model_selection,
        modelFileName=model_file_name,
        geoCenterFileName=geo_center_file,
    )

    params_dict = {
        scene.name: scene.to_step_params(),
        pre_seg_py.name: pre_seg_py.to_step_params(),
        coarse.name: coarse.to_step_params(),
        fine.name:   fine.to_step_params(),
    }
    param_start = time.time()
    client.set_params(project_id, params_dict)
    logging.info(f"Parameters set successfully in {time.time() - param_start}")

    # 3. Run vision
    play_start = time.time()
    result = client.run_vision(project_id)
    print(f"Coarse Match took {result['coarse_time_s']}")
    print(f"Fine Match took {result['fine_time_s']}")
    rp(result)

    client.close()
    print(f"total time: {time.time() - start}")