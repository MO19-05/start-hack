#!/usr/bin/env python3
"""
Simple script to test camera access via OpenCV.
Usage:
  python scripts/test_camera.py [device]
Example:
  python scripts/test_camera.py 0
"""
import sys
import cv2
import os

def main():
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('CAMERA_DEVICE', '0'))
    print(f"Opening camera device {dev}...")
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print("ERROR: Could not open camera device.")
        return 2
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: Could not read frame from camera.")
        cap.release()
        return 3
    fname = 'test_camera.jpg'
    ok = cv2.imwrite(fname, frame)
    cap.release()
    if ok:
        h, w = frame.shape[:2]
        print(f"Captured frame saved to {fname} ({w}x{h})")
        return 0
    else:
        print("ERROR: Failed to write frame to disk.")
        return 4

if __name__ == '__main__':
    sys.exit(main())

