"""Camera service that runs a GStreamer appsink pipeline in a background thread
and uses the existing ObjectDetector instance to run detection on each frame.

This keeps the detector model-loading logic in `object_detector.py` while allowing
the FastAPI backend to poll `CameraService.get_state()` and `get_frame()`.
"""
import threading
import time
from typing import Optional, Dict

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
except Exception:
    Gst = None

import cv2
import numpy as np
from .object_detector import ObjectDetector
import os


class CameraService:
    def __init__(self, device: int = 0, model_type: str = 'onnx', verbose: bool = False):
        self.device = device
        self.detector = ObjectDetector(video_device=device, verbose=verbose, model_type=model_type)
        self._thread: Optional[threading.Thread] = None
        self._loop = None
        self._running = False
        self._last_counts: Dict[str, int] = {}
        self._last_custom_counts: Dict[str, int] = {}
        self._last_frame_bytes: Optional[bytes] = None
        self._last_frame_time: Optional[float] = None
        # runtime indicates which capture backend is active: 'gstreamer' or 'opencv' or None
        self._runtime: Optional[str] = None
        self._last_yolo_boxes = []
        self._last_onnx_boxes = []

    def start(self):
        if self._running:
            return
        # Temporary image fallback: use CAMERA_FALLBACK_IMAGE env or common Windows Downloads path
        fallback = os.environ.get('CAMERA_FALLBACK_IMAGE')
        if not fallback:
            # detect common Downloads location on Windows
            userprofile = os.environ.get('USERPROFILE') or os.environ.get('HOME')
            if userprofile:
                candidate = os.path.join(userprofile, 'Downloads', 'office.jpg')
                if os.path.exists(candidate):
                    fallback = candidate

        if fallback and os.path.exists(fallback):
            self._runtime = 'image'
            self._running = True
            self._thread = threading.Thread(target=self._image_loop, args=(fallback,), daemon=True)
            self._thread.start()
            print(f"CameraService: using fallback image {fallback}")
            return

        # Prefer GStreamer when available; otherwise fall back to OpenCV capture
        if Gst is not None:
            self._runtime = 'gstreamer'
            self._running = True
            self._thread = threading.Thread(target=self._gst_loop, daemon=True)
            self._thread.start()
        else:
            # Fall back to OpenCV VideoCapture for Windows/WSL dev environments
            self._runtime = 'opencv'
            self._running = True
            self._thread = threading.Thread(target=self._opencv_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._loop:
                self._loop.quit()
        except Exception:
            pass

    def _demo_loop(self):
        # Periodically run dummy detection to populate counts
        import random
        while self._running:
            # fake frame as grey image
            frame = (np.ones((480, 640, 3), dtype=np.uint8) * 127)
            try:
                annotated, counts = self.detector.annotate_frame(frame)
                self._last_counts = dict(counts)
                # record boxes
                try:
                    self._last_yolo_boxes = list(self.detector.current_yolo_boxes)
                except Exception:
                    self._last_yolo_boxes = []
                try:
                    self._last_onnx_boxes = list(self.detector.current_onnx_boxes)
                except Exception:
                    self._last_onnx_boxes = []
                self._last_custom_counts = {}
                ok, buf = cv2.imencode('.jpg', annotated)
                if ok:
                    self._last_frame_bytes = buf.tobytes()
                else:
                    self._last_frame_bytes = None
            except Exception:
                self._last_frame_bytes = None
            time.sleep(1.0)

    def _opencv_loop(self):
        """Capture frames with OpenCV VideoCapture and run detection."""
        cap = None
        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(self.device)
            if not cap.isOpened():
                # Try without CAP_DSHOW
                cap = cv2.VideoCapture(self.device)

            # Read loop
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # small backoff if camera read fails
                    time.sleep(0.5)
                    continue

                # Ensure frame is BGR uint8
                try:
                    annotated, counts = self.detector.annotate_frame(frame)
                    self._last_counts = dict(counts)
                    # record boxes
                    try:
                        self._last_yolo_boxes = list(self.detector.current_yolo_boxes)
                    except Exception:
                        self._last_yolo_boxes = []
                    try:
                        self._last_onnx_boxes = list(self.detector.current_onnx_boxes)
                    except Exception:
                        self._last_onnx_boxes = []
                    self._last_custom_counts = {}
                    ok, buf = cv2.imencode('.jpg', annotated)
                    if ok:
                        self._last_frame_bytes = buf.tobytes()
                        self._last_frame_time = time.time()
                except Exception:
                    # keep last frame on failure
                    pass

                # throttle loop slightly to avoid maxing CPU
                time.sleep(0.03)
        except Exception as e:
            print("OpenCV capture loop error:", e)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    def _image_loop(self, path: str):
        """Load a single image from disk and repeatedly run detection on it.

        Useful for Windows/WSL testing when a real camera is not available.
        """
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"CameraService: failed to read fallback image: {path}")
                return

            while self._running:
                try:
                    annotated, counts = self.detector.annotate_frame(img)
                    self._last_counts = dict(counts)
                    # record boxes
                    try:
                        self._last_yolo_boxes = list(self.detector.current_yolo_boxes)
                    except Exception:
                        self._last_yolo_boxes = []
                    try:
                        self._last_onnx_boxes = list(self.detector.current_onnx_boxes)
                    except Exception:
                        self._last_onnx_boxes = []
                    self._last_custom_counts = {}
                    ok, buf = cv2.imencode('.jpg', annotated)
                    if ok:
                        self._last_frame_bytes = buf.tobytes()
                        self._last_frame_time = time.time()
                except Exception as e:
                    print("CameraService image loop detection error:", e)
                # update at 1 FPS to reduce CPU
                time.sleep(1.0)
        except Exception as e:
            print("CameraService image loop error:", e)

    def _gst_loop(self):
        # Build a simple appsink pipeline and handle samples in callback
        pipeline_str = (
            f"v4l2src device=/dev/video{self.device} ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self._pipeline = Gst.parse_launch(pipeline_str)
        appsink = self._pipeline.get_by_name('sink')
        appsink.connect('new-sample', self._on_new_sample)

        # Bus and loop
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()

        self._loop = GLib.MainLoop()

        # start pipeline
        self._pipeline.set_state(Gst.State.PLAYING)

        try:
            self._loop.run()
        except Exception:
            pass
        finally:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

    def _on_new_sample(self, appsink):
        sample = appsink.emit('pull-sample')
        if sample is None:
            return

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return

        frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data)
        buffer.unmap(map_info)

        # Run detector and annotate frame (use provided annotate_frame)
        try:
            annotated, counts = self.detector.annotate_frame(frame)
            self._last_counts = dict(counts)
            self._last_custom_counts = {}
            try:
                ok, buf = cv2.imencode('.jpg', annotated)
                if ok:
                    self._last_frame_bytes = buf.tobytes()
                    self._last_frame_time = time.time()
            except Exception:
                # keep previous annotated frame if encoding fails
                pass
        except Exception as e:
            # keep previous counts on error
            print("CameraService detection error:", e)

        return Gst.FlowReturn.OK

    def get_state(self) -> Dict:
        # Interpret 'person' count as number of people
        num_people = int(self._last_counts.get('person', 0) or 0)
        # Determine cleanliness heuristically: if many bottles/cups present -> needs_cleaning
        bottles = int(self._last_counts.get('bottle', 0) or 0) + int(self._last_custom_counts.get('bottle', 0) or 0)
        cups = int(self._last_counts.get('cup', 0) or 0)
        cleanliness = 'clean'
        if bottles + cups >= 3:
            cleanliness = 'needs_cleaning'
        elif bottles + cups > 0:
            cleanliness = 'in_progress'

        return {'num_people': num_people, 'cleanliness': cleanliness}

    def get_frame(self) -> Optional[bytes]:
        return self._last_frame_bytes

    def get_status(self) -> Dict:
        return {
            'running': self._running,
            'runtime': self._runtime,
            'device': self.device,
            'last_frame_time': self._last_frame_time,
            'last_frame_size': len(self._last_frame_bytes) if self._last_frame_bytes else 0,
            'last_counts': dict(self._last_counts),
            'last_yolo_boxes': self._last_yolo_boxes,
            'last_onnx_boxes': self._last_onnx_boxes,
        }


# Singleton service instance
_service: Optional[CameraService] = None


def start_camera_service(device: int = 0, model_type: str = 'onnx', verbose: bool = False):
    global _service
    if _service is None:
        _service = CameraService(device=device, model_type=model_type, verbose=verbose)
        _service.start()
    return _service


def get_camera_service() -> Optional[CameraService]:
    return _service

