import argparse
import sys
import time
import os
from picamera2 import Picamera2

def capture_image(camera_index, output_filename):
    """
    Initializes the specific camera, warms it up, and saves a snapshot.
    """
    try:
        # Initialize Picamera2 with the specific camera index
        # On RPi5: index 0 is usually CAM/DISP 0, index 1 is CAM/DISP 1
        picam2 = Picamera2(camera_index)
        
        # Configure the camera for a still capture
        # This automatically handles the specific tuning for IMX219 or OV5647
        config = picam2.create_still_configuration(main={"size": (640, 480) if camera_index == 0 else (640, 480)}) 
        picam2.configure(config)
        
        print(f"[*] Starting Camera {camera_index}...")
        picam2.start()
        
        # Warmup time is crucial for Auto White Balance (AWB) and Auto Exposure (AE) to settle
        print("[*] Warming up sensor (2 seconds)...")
        time.sleep(2)
        
        # Capture the image
        print(f"[*] Capturing to {output_filename}...")
        picam2.capture_file(output_filename)
        
        # Stop the camera and release resources
        picam2.stop()
        print("[*] Done.")

    except RuntimeError as e:
        print(f"\n[!] Error: Could not access Camera {camera_index}.")
        print(f"    Details: {e}")
        print("    Tip: Run 'rpicam-hello --list-cameras' to verify your indices.")
        sys.exit(1)
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Capture dataset images on RPi5 with Dual Cameras")
    
    # Argument to select the camera
    parser.add_argument(
        '-c', '--camera', 
        type=int, 
        choices=[1, 2], 
        required=True, 
        help="Select which camera to use: 1 or 2"
    )
    
    # Argument for the output filename
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True, 
        help="The output filename (e.g., dataset_01.jpg)"
    )

    args = parser.parse_args()

    # Ensure output directory exists if provided in path
    directory = os.path.dirname(args.output)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # MAP USER INPUT TO SYSTEM INDEX
    # Note: Hardware assignment can vary. 
    # Usually, the port labeled "CAM/DISP 0" is index 0.
    # We map user input (1 or 2) to system index (0 or 1).
    system_index = args.camera - 1

    capture_image(system_index, args.output)

if __name__ == "__main__":
    main()
