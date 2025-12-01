import threading
import time
from typing import Dict, Any, Optional

# Simple in-memory store for frames and states sent by edge devices (Raspberry Pi)
_lock = threading.Lock()
# store structure: room_id -> { 'frame': bytes, 'counts': dict, 'boxes': list, 'ts': float }
_STORE: Dict[str, Dict[str, Any]] = {}


def set_frame(room_id: str, frame_bytes: bytes, counts: Dict[str, int], boxes: list, ts: Optional[float] = None):
    if ts is None:
        ts = time.time()
    with _lock:
        _STORE[room_id] = {
            'frame': frame_bytes,
            'counts': counts or {},
            'boxes': boxes or [],
            'ts': ts,
        }


def get_frame(room_id: str) -> Optional[bytes]:
    with _lock:
        entry = _STORE.get(room_id)
        if not entry:
            return None
        return entry.get('frame')


def get_state(room_id: str) -> Dict[str, Any]:
    with _lock:
        entry = _STORE.get(room_id)
        if not entry:
            return {}
        return {
            'num_people': int(entry.get('counts', {}).get('person', 0)),
            'cleanliness': entry.get('counts', {}).get('cleanliness', 'clean'),
            'counts': entry.get('counts', {}),
            'boxes': entry.get('boxes', []),
            'ts': entry.get('ts')
        }


def get_status(room_id: str) -> Dict[str, Any]:
    with _lock:
        entry = _STORE.get(room_id)
        if not entry:
            return {'ok': False}
        return {'ok': True, 'ts': entry.get('ts'), 'counts': entry.get('counts'), 'boxes': entry.get('boxes')}

