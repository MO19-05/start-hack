#!/usr/bin/env python3
"""Quick check if bounding boxes are present in the detected image"""
import cv2
import sys

# Load both images
original = cv2.imread('yfood_image_21.jpg')
detected = cv2.imread('yfood_image_21_detected.jpg')

if original is None or detected is None:
    print("ERROR: Could not load images")
    sys.exit(1)

# Check if they're different (boxes were drawn)
diff = cv2.absdiff(original, detected)
diff_sum = diff.sum()

print(f"Original image shape: {original.shape}")
print(f"Detected image shape: {detected.shape}")
print(f"Pixel difference sum: {diff_sum}")

if diff_sum > 1000000:  # Arbitrary threshold
    print("\n[OK] SUCCESS: Bounding boxes appear to be drawn (significant pixel differences detected)")
else:
    print("\n[WARN] WARNING: Images look very similar (bounding boxes may not be visible)")

# Find regions with significant changes (where boxes likely are)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    print(f"\nFound {len(contours)} regions with changes:")
    for i, cnt in enumerate(contours[:5]):  # Show first 5
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"  Region {i+1}: x={x}, y={y}, w={w}, h={h}")

