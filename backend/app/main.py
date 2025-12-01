from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import Dict, Any, List
import asyncio
import io
import time
from . import inference
from . import edge_store
from fastapi import UploadFile, File, Form, Request
import os
from . import notify
from .config import settings

app = FastAPI(title="Room Monitoring Backend")

from .config import settings

# Configure CORS origins. Use BACKEND_ALLOW_ORIGINS env var (comma-separated) if provided,
# otherwise allow common localhost dev origins (Vite default 5173 and alternative 8080).
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Allow opt-in wildcard for quick dev/debugging via ALLOW_ALL_ORIGINS=true
import os as _os
_allow_all = str(_os.environ.get("ALLOW_ALL_ORIGINS", "")).lower() in ("1", "true", "yes")
raw = _os.environ.get("BACKEND_ALLOW_ORIGINS")
if _allow_all:
    # Dev mode: allow any origin (note: allow_credentials will be disabled for safety)
    allow_origins = ["*"]
    allow_credentials = False
    print("WARNING: ALLOW_ALL_ORIGINS is enabled — CORS allows any origin (dev only). allow_credentials=False")
else:
    if raw:
        # split comma-separated list
        allow_origins = [o.strip() for o in raw.split(",") if o.strip()]
    else:
        allow_origins = _default_origins
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of room capacities and last-known state
ROOM_CAPACITIES = {
    "101": 12,
    "102": 16,
    "103": 8,
    "104": 20,
    "105": 6,
    "106": 24,
}

class RoomState(BaseModel):
    room_id: str
    num_people: int
    cleanliness: str
    capacity: int
    warning: bool = False


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def broadcast(self, message: Dict[str, Any]):
        to_remove = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                to_remove.append(connection)
        for c in to_remove:
            self.disconnect(c)


manager = ConnectionManager()

# latest state cache
LATEST_STATES: Dict[str, RoomState] = {}


@app.on_event("startup")
async def startup_event():
    # initialize state
    for rid, cap in ROOM_CAPACITIES.items():
        LATEST_STATES[rid] = RoomState(room_id=rid, num_people=0, cleanliness="clean", capacity=cap)
    # Optionally start camera service if configured to map a room to the local camera
    camera_room = os.environ.get("CAMERA_ROOM_ID", "101")
    camera_device = int(os.environ.get("CAMERA_DEVICE", "0"))
    if camera_room in ROOM_CAPACITIES:
        try:
            from .camera_service import start_camera_service
            # start camera for the mapped room using ONNX models
            start_camera_service(device=camera_device, model_type='onnx', verbose=False)
            print(f"Started camera for room {camera_room} on /dev/video{camera_device}")
        except Exception as e:
            print("Failed to start camera service:", e)
            # Fail startup so the system doesn't run without camera support
            raise
    else:
        print(f"No camera mapping configured for room {camera_room}; backend will run without local camera.")

    # start periodic inference loop
    asyncio.create_task(periodic_inference_loop())


async def periodic_inference_loop():
    while True:
        try:
            await run_inference_for_all_rooms()
        except Exception as e:
            print("Error during periodic inference:", e)
        await asyncio.sleep(30)


async def run_inference_for_all_rooms():
    warnings: List[Dict[str, Any]] = []
    for room_id, cap in ROOM_CAPACITIES.items():
        try:
            state = inference.get_room_state(room_id)
            num_people = int(state.get("num_people", 0))
            cleanliness = state.get("cleanliness", "clean")
        except Exception as e:
            print("Inference error for room", room_id, e)
            continue

        warn = False
        if cap > 0 and num_people / cap >= 0.8:
            warn = True
        if cleanliness == "needs_cleaning":
            warn = True

        room_state = RoomState(room_id=room_id, num_people=num_people, cleanliness=cleanliness, capacity=cap, warning=warn)
        LATEST_STATES[room_id] = room_state

        if warn:
            warnings.append({"room_id": room_id, "num_people": num_people, "cleanliness": cleanliness, "capacity": cap})

    # broadcast warnings if any
    if warnings:
        payload = {"type": "warning", "warnings": warnings, "ts": time.time()}
        await manager.broadcast(payload)
        # send email notification summary asynchronously if SMTP configured
        try:
            subject = f"Room warnings detected ({len(warnings)})"
            body_lines = [f"Room {w['room_id']}: {w['num_people']}/{w['capacity']} people, cleanliness={w['cleanliness']}" for w in warnings]
            body = "\n".join(body_lines)
            # attempt async send
            asyncio.create_task(notify.send_email_async(subject, body))
        except Exception as e:
            print("Error scheduling email send:", e)


@app.get("/rooms")
async def list_rooms():
    return [s.dict() for s in LATEST_STATES.values()]


@app.get("/rooms/{room_id}/state")
async def get_room_state(room_id: str):
    if room_id not in LATEST_STATES:
        raise HTTPException(status_code=404, detail="Room not found")
    return LATEST_STATES[room_id].dict()


@app.get("/rooms/{room_id}/frame")
async def get_room_frame(room_id: str):
    # Call inference to retrieve frame bytes (JPEG/PNG)
    try:
        frame_bytes = inference.get_room_frame(room_id)
    except Exception as e:
        print("Frame inference error:", e)
        frame_bytes = None

    if not frame_bytes:
        # Return 204 No Content so frontend can fall back
        return Response(status_code=204)

    return StreamingResponse(io.BytesIO(frame_bytes), media_type="image/jpeg")


@app.get("/rooms/{room_id}/stream")
async def stream_room_frames_endpoint(room_id: str):
    """Return an MJPEG (multipart/x-mixed-replace) stream for the room.

    The stream polls `inference.get_room_frame(room_id)` on a short interval
    and yields JPEG frames as multipart chunks. Configure interval with
    `STREAM_INTERVAL_MS` env var (default 200 ms).
    """
    boundary = "frame"
    interval_ms = int(os.environ.get("STREAM_INTERVAL_MS", "200"))

    async def generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                # call potentially-blocking inference.get_room_frame in threadpool
                frame_bytes = await loop.run_in_executor(None, inference.get_room_frame, room_id)
                if frame_bytes:
                    header = (f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame_bytes)}\r\n\r\n").encode("utf-8")
                    yield header + frame_bytes + b"\r\n"
            except Exception as e:
                print("stream_room_frames: error retrieving frame:", e)
            await asyncio.sleep(interval_ms / 1000.0)

    return StreamingResponse(generator(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@app.get("/camera/status")
async def camera_status():
    try:
        from .camera_service import get_camera_service
        svc = get_camera_service()
        if not svc:
            return {"ok": False, "reason": "camera service not started"}
        return {"ok": True, "status": svc.get_status()}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


@app.post("/edge/rooms/{room_id}/frame")
async def ingest_edge_frame(room_id: str, request: Request, image: UploadFile = File(None), counts: str = Form(None), boxes: str = Form(None), ts: float = Form(None)):
    """Endpoint for edge clients (Raspberry Pi) to POST annotated frames and metadata.

    Accepts multipart/form-data with fields:
    - `image` (bytes, required): annotated JPEG bytes
    - `counts` (JSON string, required): e.g. '{"person": 3, "bottle": 1}'
    - `boxes` (JSON string, optional): list of boxes
    - `ts` (float, optional): timestamp
    """
    import json
    import base64

    # Read raw request and support both multipart/form-data (preferred) and
    # application/json (fallback with base64-encoded image). We intentionally
    # avoid declaring Form/UploadFile parameters in the signature so FastAPI
    # doesn't validate the body before our handler runs (which caused 422s).
    content_type = request.headers.get('content-type', '')
    image_bytes = None
    counts_obj = {}
    boxes_obj = []
    ts_val = None

    try:
        if content_type and 'multipart/form-data' in content_type:
            form = await request.form()
            # 'image' may be an UploadFile
            image_field = form.get('image')
            if image_field is not None:
                try:
                    # UploadFile provided by Starlette has .read()
                    image_bytes = await image_field.read()
                except Exception:
                    # fallback: maybe it's already bytes-like
                    try:
                        image_bytes = image_field.file.read()
                    except Exception:
                        image_bytes = None

            counts_raw = form.get('counts')
            boxes_raw = form.get('boxes')
            ts_raw = form.get('ts')
            try:
                counts_obj = json.loads(counts_raw) if counts_raw else {}
            except Exception:
                counts_obj = {}
            try:
                boxes_obj = json.loads(boxes_raw) if boxes_raw else []
            except Exception:
                boxes_obj = []
            try:
                ts_val = float(ts_raw) if ts_raw else None
            except Exception:
                ts_val = None

        elif content_type and 'application/json' in content_type:
            body = await request.json()
            # Expect JSON like: {"image": "<base64>", "counts": {...}, "boxes": [...], "ts": 123}
            img_b64 = body.get('image')
            if img_b64:
                try:
                    image_bytes = base64.b64decode(img_b64)
                except Exception:
                    image_bytes = None
            counts_obj = body.get('counts') or {}
            boxes_obj = body.get('boxes') or []
            ts_val = body.get('ts')
        else:
            # Unknown/unsupported content type—try to parse form anyway to provide
            # a clearer error message instead of leaving FastAPI to return 422.
            try:
                form = await request.form()
                image_field = form.get('image')
                if image_field is not None:
                    try:
                        image_bytes = await image_field.read()
                    except Exception:
                        try:
                            image_bytes = image_field.file.read()
                        except Exception:
                            image_bytes = None
                counts_raw = form.get('counts')
                try:
                    counts_obj = json.loads(counts_raw) if counts_raw else {}
                except Exception:
                    counts_obj = {}
            except Exception:
                print(f"ingest_edge_frame: unsupported content-type={content_type}")
                return JSONResponse(status_code=400, content={"ok": False, "reason": "Unsupported Content-Type. Use multipart/form-data with field 'image' or application/json with base64 'image'."})
    except Exception as e:
        print("ingest_edge_frame: exception parsing request:", e)
        return JSONResponse(status_code=400, content={"ok": False, "reason": "Failed to parse request body."})

    # Store into edge store
    try:
        edge_store.set_frame(room_id, image_bytes, counts_obj, boxes_obj, float(ts_val) if ts_val else None)
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    # Log stored frame info for debugging
    try:
        size = len(image_bytes) if image_bytes else 0
    except Exception:
        size = 0
    print(f"ingest_edge_frame: stored frame for room={room_id} size={size} counts={counts_obj} ts={ts_val}")

    # Also update LATEST_STATES for quick frontend access
    cap = ROOM_CAPACITIES.get(room_id, 0)
    num_people = int(counts_obj.get('person', 0))
    cleanliness = counts_obj.get('cleanliness', 'clean')
    warn = False
    if cap > 0 and num_people / cap >= 0.8:
        warn = True
    if cleanliness == 'needs_cleaning':
        warn = True
    LATEST_STATES[room_id] = RoomState(room_id=room_id, num_people=num_people, cleanliness=cleanliness, capacity=cap, warning=warn)

    return {"ok": True}


@app.get("/edge/rooms/{room_id}/status")
async def edge_room_status(room_id: str):
    try:
        st = edge_store.get_status(room_id)
        if not st.get('ok'):
            return JSONResponse(status_code=404, content={"ok": False, "reason": "no frame for room"})
        return {"ok": True, "ts": st.get('ts'), "counts": st.get('counts'), "boxes": st.get('boxes')}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "reason": str(e)})



class AlertIn(BaseModel):
    message: str


@app.post("/rooms/{room_id}/alert")
async def send_alert(room_id: str, alert: AlertIn, background_tasks: BackgroundTasks):
    # For demo: broadcast alert to connected clients and optionally trigger email (not implemented)
    payload = {"type": "alert", "room_id": room_id, "message": alert.message, "ts": time.time()}
    background_tasks.add_task(manager.broadcast, payload)
    # Also send email to configured recipients (blocking send run in background task)
    try:
        subject = f"Alert for room {room_id}"
        body = f"An alert was sent for room {room_id}:\n\n{alert.message}"
        background_tasks.add_task(notify.send_email_sync, subject, body, None)
    except Exception as e:
        print("Failed to queue email send:", e)

    return {"ok": True}


@app.websocket("/ws/notifications")
async def websocket_notifications(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # keepalive / optionally receive pings from client
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

