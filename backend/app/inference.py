"""Placeholder inference wrappers.
These call the actual inference implementation (assumed to be provided elsewhere).
If that module is unavailable, fall back to a simulated response so the backend remains runnable.
"""
from typing import Dict, Tuple, Optional
import random
import os
import glob
import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ONNXRT = True
except Exception:
    HAS_ONNXRT = False

try:
    # Real implementation (provided by another developer) should expose these functions
    from real_inference import get_room_state as real_get_room_state, get_room_frame as real_get_room_frame
    HAS_REAL = True
except Exception:
    HAS_REAL = False

try:
    # Use camera service if available
    from .camera_service import get_camera_service, start_camera_service
    HAS_CAMERA_SERVICE = True
except Exception:
    HAS_CAMERA_SERVICE = False
try:
    from .edge_store import get_state as edge_get_state, get_frame as edge_get_frame
    HAS_EDGE_STORE = True
except Exception:
    HAS_EDGE_STORE = False


# Try to locate a people-detection ONNX model. Priority:
# 1. env PEOPLE_ONNX_PATH
# 2. any file matching 'People_*.onnx' inside this package directory
PEOPLE_ONNX_PATH = os.environ.get("PEOPLE_ONNX_PATH")
_module_dir = os.path.dirname(__file__)
if not PEOPLE_ONNX_PATH:
    # search for People_*.onnx in module dir
    matches = glob.glob(os.path.join(_module_dir, "People_*.onnx"))
    if matches:
        PEOPLE_ONNX_PATH = matches[0]

_people_session = None
_people_input = None
if PEOPLE_ONNX_PATH and HAS_ONNXRT:
    try:
        _people_session = ort.InferenceSession(PEOPLE_ONNX_PATH, providers=["CPUExecutionProvider"])
        _people_input = _people_session.get_inputs()[0]
        print(f"Loaded people ONNX model: {PEOPLE_ONNX_PATH}")
    except Exception as e:
        print("Failed to load people ONNX model:", e)
        _people_session = None

# Which room is backed by the camera on this host (single-camera mapping)
CAMERA_ROOM_ID = os.environ.get("CAMERA_ROOM_ID", "101")


def _infer_people_from_bytes(frame_bytes: bytes) -> int:
    """Decode JPEG bytes and run people ONNX model session. Return estimated number of people (int).
    Uses simple heuristics to count detections from model outputs.
    """
    if _people_session is None:
        return 0
    try:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0

        # Prepare input tensor according to model input shape if available
        input_shape = None
        try:
            shape = _people_input.shape  # e.g. [1,3,224,224] or [None,3,256,256]
            # determine height/width from shape
            # shape can be NCHW ([N,C,H,W]) or NHWC ([N,H,W,C]).
            # Determine channel position: if shape[1] is 1 or 3, assume NCHW; otherwise if last dim is 1 or 3 assume NHWC.
            if len(shape) >= 4:
                try:
                    s1 = int(shape[1]) if shape[1] else None
                except Exception:
                    s1 = None
                try:
                    slast = int(shape[-1]) if shape[-1] else None
                except Exception:
                    slast = None

                if s1 in (1, 3):
                    # NCHW: [N, C, H, W]
                    h = int(shape[2]) if shape[2] else None
                    w = int(shape[3]) if shape[3] else None
                    if h and w:
                        input_shape = (w, h)
                        expect_nchw = True
                    else:
                        input_shape = None
                        expect_nchw = True
                elif slast in (1, 3):
                    # NHWC: [N, H, W, C]
                    h = int(shape[1]) if shape[1] else None
                    w = int(shape[2]) if shape[2] else None
                    if h and w:
                        input_shape = (w, h)
                        expect_nchw = False
                    else:
                        input_shape = None
                        expect_nchw = False
                else:
                    # Fallback: try treating as NHWC
                    try:
                        h = int(shape[1]) if shape[1] else None
                        w = int(shape[2]) if shape[2] else None
                        if h and w:
                            input_shape = (w, h)
                            expect_nchw = False
                        else:
                            input_shape = None
                            expect_nchw = True
                    except Exception:
                        input_shape = None
                        expect_nchw = True
        except Exception:
            input_shape = None
            # default assume NCHW if not determined
            if 'expect_nchw' not in locals():
                expect_nchw = True

        if input_shape:
            img_resized = cv2.resize(img, input_shape)
        else:
            img_resized = img

        # Normalize to 0-1 and convert to float32
        img_in = img_resized.astype(np.float32) / 255.0

        # Prepare tensor according to expected layout
        if expect_nchw:
            # transpose HWC->CHW
            img_in = np.transpose(img_in, (2, 0, 1))
            img_in = np.expand_dims(img_in, axis=0).astype(np.float32)
        else:
            # NHWC: just add batch dim
            img_in = np.expand_dims(img_in, axis=0).astype(np.float32)

        outputs = _people_session.run(None, { _people_input.name: img_in })

        # Heuristic: find confidences array in outputs and count values > 0.5
        count = 0
        for out in outputs:
            try:
                a = np.array(out)
                # flatten and threshold
                if a.dtype.kind in ('f', 'i'):
                    vals = a.flatten()
                    # normalize if values appear large
                    if vals.max() > 1.5:
                        # maybe raw logits; apply sigmoid
                        vals = 1 / (1 + np.exp(-vals))
                    # count confident detections
                    c = int((vals > 0.5).sum())
                    if c > count:
                        count = c
            except Exception:
                continue

        return int(count)
    except Exception as e:
        print("People model inference error:", e)
        return 0


def get_room_state(room_id: str) -> Dict:
    """Return a dict with keys: num_people (int), cleanliness (str: 'clean'|'needs_cleaning'|'in_progress').
    Priority: real_inference -> camera_service -> simulated."""
    # Real dedicated inference implementation wins
    if HAS_REAL:
        return real_get_room_state(room_id)

    # If a camera service is running and this room is mapped to the local camera, use it
    # Edge (Raspberry Pi) uploads take priority if available for this room
    if HAS_EDGE_STORE:
        try:
            st = edge_get_state(room_id)
            if st:
                return st
        except Exception:
            pass

    if room_id == CAMERA_ROOM_ID and HAS_CAMERA_SERVICE:
        svc = get_camera_service()
        if svc:
            try:
                # If a dedicated people ONNX model is available, prefer running it on the latest frame
                if _people_session is not None:
                    frame_bytes = svc.get_frame()
                    if frame_bytes:
                        num_people = _infer_people_from_bytes(frame_bytes)
                        # reuse camera-service cleanliness heuristic
                        state = svc.get_state()
                        cleanliness = state.get("cleanliness", "clean")
                        return {"num_people": int(num_people), "cleanliness": cleanliness}

                return svc.get_state()
            except Exception:
                pass

    # Simulated values for development/demo
    num_people = random.randint(0, 20)
    cleanliness = random.choices(["clean", "needs_cleaning", "in_progress"], weights=[0.7, 0.2, 0.1])[0]
    return {"num_people": num_people, "cleanliness": cleanliness}


def get_room_frame(room_id: str) -> Optional[bytes]:
    """Return raw JPEG/PNG bytes for the frame with detection overlay.
    Return None when not available.
    """
    if HAS_REAL:
        return real_get_room_frame(room_id)

    # First prefer edge-store frames
    if HAS_EDGE_STORE:
        try:
            fb = edge_get_frame(room_id)
            if fb:
                return fb
        except Exception:
            pass

    # Only return camera frames for the mapped camera room
    if room_id == CAMERA_ROOM_ID and HAS_CAMERA_SERVICE:
        svc = get_camera_service()
        if svc:
            try:
                return svc.get_frame()
            except Exception:
                return None

    # No real camera available; return None so caller can degrade gracefully
    return None

