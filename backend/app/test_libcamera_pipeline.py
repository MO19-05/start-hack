#!/usr/bin/env python3
"""Quick test of libcamerasrc pipeline"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys

Gst.init(None)

camera_name = "/base/axi/pcie@1000120000/rp1/i2c@88000/imx219@10"

# Test pipeline - minimal, let everything autonegotiate
pipeline_str = f'libcamerasrc camera-name="{camera_name}" ! videoconvert ! autovideosink'

print(f"Testing pipeline:\n{pipeline_str}\n")

try:
    pipeline = Gst.parse_launch(pipeline_str)
    print("[OK] Pipeline created successfully")
    
    # Add bus watch for errors
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] {err}")
            print(f"   Debug: {debug}")
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == pipeline:
                old, new, pending = message.parse_state_changed()
                print(f"Pipeline state: {old.value_nick} -> {new.value_nick}")
        return True
    
    bus.connect("message", on_message)
    
    # Start pipeline
    print("\nStarting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline")
        # Try to get error message
        msg = bus.timed_pop_filtered(1000000000, Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            print(f"   Error: {err}")
            print(f"   Debug: {debug}")
        sys.exit(1)
    elif ret == Gst.StateChangeReturn.ASYNC:
        print("Pipeline starting (async)...")
    else:
        print("[OK] Pipeline started")
    
    # Run for 5 seconds
    print("\nRunning for 5 seconds (press Ctrl+C to stop earlier)...")
    loop = GLib.MainLoop()
    
    def timeout_cb():
        print("\n[OK] Test completed successfully")
        loop.quit()
        return False
    
    GLib.timeout_add_seconds(5, timeout_cb)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted")
    
    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    print("[OK] Pipeline stopped")
    
except Exception as e:
    print(f"[ERROR] Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

