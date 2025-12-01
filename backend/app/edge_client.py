#!/usr/bin/env python3
"""Edge client to run on Raspberry Pi.

Captures frames (OpenCV), runs the ONNX detectors via ObjectDetector, draws boxes,
and uploads annotated JPEG plus counts/boxes to the backend server.

Usage (on Raspberry Pi):

export EDGE_BACKEND_URL=http://<backend-host>:8000
export ROOM_ID=101
python3 app/edge_client.py

The script expects the ONNX models to be available in the same folder as object_detector.py
(or set PEOPLE_ONNX_PATH / BOTTLE_ONNX_PATH).
"""
import os
import time
import json
import traceback
import requests
import cv2
from object_detector import ObjectDetector

EDGE_BACKEND_URL = os.environ.get('EDGE_BACKEND_URL', 'https://f5a7912ee4e3.ngrok-free.app')
ROOM_ID = os.environ.get('ROOM_ID', os.environ.get('CAMERA_ROOM_ID', '101'))
INTERVAL = float(os.environ.get('EDGE_PUBLISH_INTERVAL', '5.0'))
CAMERA_DEVICE = int(os.environ.get('CAMERA_DEVICE', '0'))
VERBOSE = os.environ.get('EDGE_VERBOSE', '0') in ('1','true','yes')


def main():
    detector = ObjectDetector(video_device=CAMERA_DEVICE, verbose=VERBOSE, model_type='onnx', mode='detect')

    # Use OpenCV VideoCapture for simple camera capture (works well on Raspberry Pi with /dev/video0)
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"Failed to open camera device {CAMERA_DEVICE}")
        return

    print(f"Edge client started. Sending frames to {EDGE_BACKEND_URL}/edge/rooms/{ROOM_ID}/frame every {INTERVAL}s")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                if VERBOSE:
                    print("Camera read failed; retrying")
                time.sleep(1.0)
                continue

            # Run detection + annotation
            annotated, counts = detector.annotate_frame(frame)

            # Prepare boxes (use detector's current boxes)
            boxes = []
            boxes.extend(detector.current_onnx_boxes or [])
            boxes.extend(detector.current_yolo_boxes or [])

            # Encode annotated image as JPEG
            success, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                if VERBOSE:
                    print("Failed to encode annotated frame")
                continue
            img_bytes = buf.tobytes()

            # Prepare multipart upload
            url = EDGE_BACKEND_URL.rstrip('/') + f"/edge/rooms/{ROOM_ID}/frame"
            files = {
                'image': ('frame.jpg', img_bytes, 'image/jpeg')
            }
            data = {
                'counts': json.dumps(counts),
                'boxes': json.dumps(boxes),
                'ts': str(time.time())
            }

            try:
                r = requests.post(url, files=files, data=data, timeout=10)
                if r.status_code == 200:
                    if VERBOSE:
                        print(f"Uploaded frame: people={counts.get('person',0)}, bottles={counts.get('bottle',0)}")
                else:
                    print(f"Upload failed: {r.status_code} {r.text}")
            except Exception as e:
                if VERBOSE:
                    print("Upload exception:", e)

            time.sleep(INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            time.sleep(1.0)

    cap.release()


if __name__ == '__main__':
    main()
