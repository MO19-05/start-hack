#!/usr/bin/env python3
"""
GStreamer pipeline for object detection
Captures webcam frames, runs object detection model, and outputs counts to terminal
"""

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
    HAS_GST = True
except Exception:
    # GStreamer / PyGObject is optional for environments that only need
    # ONNX or image-file processing (e.g., Windows / WSL development).
    Gst = None
    GLib = None
    HAS_GST = False
import numpy as np
import cv2
from collections import defaultdict
import time
import os
import onnxruntime as ort
import threading
import queue

# Ensure printing Unicode works even if the environment stdout encoding is latin-1
# Wrap sys.stdout/sys.stderr so any UnicodeEncodeError is handled by falling back
# to UTF-8 with replacement (avoids crashes on characters like ✓, emojis, etc.).
import sys

def _safe_write(s, stream):
    try:
        stream.write(s)
    except UnicodeEncodeError:
        # Encode to UTF-8 then decode using the stream encoding with replacement
        enc = getattr(stream, 'encoding', None) or 'utf-8'
        stream.write(s.encode('utf-8', errors='replace').decode(enc, errors='replace'))
    except Exception:
        # Last resort: ensure something is written
        try:
            stream.write(s.encode('utf-8', errors='replace').decode('latin-1', errors='replace'))
        except Exception:
            pass

class _StdWrapper:
    def __init__(self, stream):
        self._stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')
    def write(self, s):
        _safe_write(s, self._stream)
    def writelines(self, lines):
        for l in lines:
            self.write(l)
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
    def __getattr__(self, name):
        return getattr(self._stream, name)

# Replace stdout/stderr with wrappers so print() won't raise on encode errors
try:
    sys.stdout = _StdWrapper(sys.stdout)
    sys.stderr = _StdWrapper(sys.stderr)
except Exception:
    # If wrapping fails for any reason, continue without crashing
    pass

# Initialize GStreamer
Gst.init(None)


class ObjectDetector:
    def __init__(self, video_device=0, verbose=False, model_type='yolo', compare_models=False, 
                 custom_model_file=None, mode='detect', input_source=None):
        self.capture_pipeline = None
        self.display_pipeline = None
        self.loop = None
        self.object_counts = defaultdict(int)
        self.custom_counts = defaultdict(int)
        self.frame_count = 0
        self.last_print_time = time.time()
        self.video_device = video_device
        self.input_source = input_source
        self.is_image_input = False
        self.verbose = verbose
        self.custom_model_file = custom_model_file
        self.compare_models = compare_models
        self.mode = mode
        self.total_detections_ever = 0
        self.recording_file = None
        self.current_yolo_boxes = []
        self.current_onnx_boxes = []
        self.appsrc = None
        self.display_width = 640
        self.display_height = 480
        self.current_display_frame = None
        self.frame_lock = threading.Lock()
        self.last_annotated_frame = None
        
        # Validate mode
        valid_modes = ['detect', 'detect-display', 'display', 'record', 'record-display', 'record-detect-display']
        if mode not in valid_modes:
            print(f"Invalid mode: {mode}. Using 'detect' mode.")
            self.mode = 'detect'
        
        # Auto-detect model type if custom file provided
        if custom_model_file:
            if custom_model_file.endswith('.onnx'):
                model_type = 'onnx'
                print(f"Auto-detected ONNX model: {custom_model_file}")
            else:
                print(f"Unsupported model format: {custom_model_file}")
                print("  Supported formats: .onnx")
        
        self.model_type = model_type
        
        # Disable YOLO; this detector will use only ONNX models (people + bottle)
        self.yolo_loaded = False
        self.classes = ["person", "bottle", "cup", "laptop", "cell phone", "book"]
        
        # Load two ONNX models: people detector and bottle detector
        self.people_session = None
        self.people_input = None
        self.bottle_session = None
        self.bottle_input = None
        
        # For custom model support
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_output_name = None

        # Resolve paths from env or common filenames
        people_path = os.environ.get('PEOPLE_ONNX_PATH')
        bottle_path = os.environ.get('BOTTLE_ONNX_PATH')
        
        # If custom model file provided, load it as the primary ONNX session
        if custom_model_file and os.path.exists(custom_model_file):
            try:
                self.onnx_session = ort.InferenceSession(custom_model_file, providers=['CPUExecutionProvider'])
                onnx_input = self.onnx_session.get_inputs()[0]
                self.onnx_input_name = onnx_input.name
                onnx_output = self.onnx_session.get_outputs()[0]
                self.onnx_output_name = onnx_output.name
                print(f"Loaded custom ONNX model: {custom_model_file}")
                print(f"  Input: {self.onnx_input_name}, shape: {onnx_input.shape}")
                print(f"  Output: {self.onnx_output_name}, shape: {onnx_output.shape}")
                if self.verbose:
                    try:
                        print(f"  Providers: {self.onnx_session.get_providers()}")
                        print(f"  All outputs: {[o.name + ' ' + str(o.shape) for o in self.onnx_session.get_outputs()]}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Could not load custom ONNX model: {e}")
                import traceback
                traceback.print_exc()
        
        # fallback search in module dir for people/bottle models
        pkg_dir = os.path.dirname(__file__)
        if not people_path:
            import glob
            matches = glob.glob(os.path.join(pkg_dir, 'People_*.onnx'))
            if matches:
                people_path = matches[0]
        if not bottle_path:
            import glob
            matches = glob.glob(os.path.join(pkg_dir, 'bottle_*.onnx')) + glob.glob(os.path.join(pkg_dir, 'bottle*.onnx'))
            if matches:
                bottle_path = matches[0]

        if people_path:
            try:
                self.people_session = ort.InferenceSession(people_path, providers=['CPUExecutionProvider'])
                self.people_input = self.people_session.get_inputs()[0]
                print(f"Loaded people ONNX model: {people_path}")
                print(f"  People input shape: {self.people_input.shape}")
                if self.verbose:
                    try:
                        print(f"  People input name: {self.people_input.name}")
                        print(f"  People providers: {self.people_session.get_providers()}")
                        print(f"  People outputs: {[o.shape for o in self.people_session.get_outputs()]}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Could not load people ONNX model: {e}")

        if bottle_path:
            try:
                self.bottle_session = ort.InferenceSession(bottle_path, providers=['CPUExecutionProvider'])
                self.bottle_input = self.bottle_session.get_inputs()[0]
                print(f"Loaded bottle ONNX model: {bottle_path}")
                print(f"  Bottle input shape: {self.bottle_input.shape}")
                if self.verbose:
                    try:
                        print(f"  Bottle input name: {self.bottle_input.name}")
                        print(f"  Bottle providers: {self.bottle_session.get_providers()}")
                        print(f"  Bottle outputs: {[o.shape for o in self.bottle_session.get_outputs()]}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Could not load bottle ONNX model: {e}")

        self.onnx_loaded = bool(self.people_session or self.bottle_session or self.onnx_session)
        
        self.model_loaded = self.yolo_loaded or self.onnx_loaded
        
        if not self.model_loaded:
            print("Running in demo mode with mock detections")
            self.classes = ["person", "bottle", "cup", "laptop", "cell phone", "book"]
    
    def create_pipelines(self):
        """Create GStreamer pipelines based on mode"""
        from datetime import datetime
        
        # Base source - use file if provided, otherwise camera
        if self.input_source:
            # Video file input
            base = f"filesrc location={self.input_source} ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480"
        else:
            # Camera input - use libcamerasrc with specific camera device
            # This fixes buffer allocation issues with v4l2src on Raspberry Pi
            # Default camera path for IMX219 sensor on Raspberry Pi 5
            libcam_name = os.environ.get('LIBCAMERA_NAME', '/base/axi/pcie@1000120000/rp1/i2c@88000/imx219@10')
            # libcamerasrc needs RGBx format specified, then convert to BGR
            base = f'libcamerasrc camera-name="{libcam_name}" ! video/x-raw,format=RGBx,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR'
            if self.verbose:
                print(f"Using libcamerasrc with camera: {libcam_name}")
        
        # Build pipelines based on mode
        if self.mode == 'detect':
            # Detection only: webcam -> appsink
            # Insert a small queue before appsink to reduce buffer-pool pressure
            capture_str = (
                f"{base} ! "
                "queue max-size-buffers=2 leaky=downstream ! "
                "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
            )
            if self.verbose:
                print(f"\nCapture pipeline string:\n{capture_str}\n")
            self.capture_pipeline = Gst.parse_launch(capture_str)
        
        elif self.mode == 'detect-display':
            # Capture pipeline: webcam -> appsink
            capture_str = (
                f"{base} ! "
                "queue max-size-buffers=2 leaky=downstream ! "
                "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true"
            )
            self.capture_pipeline = Gst.parse_launch(capture_str)
            
            # Display pipeline: appsrc -> display
            display_str = (
                "appsrc name=src format=time is-live=true do-timestamp=true block=false max-bytes=0 "
                "! video/x-raw,format=BGR,width=640,height=480,framerate=30/1 "
                "! queue max-size-buffers=2 leaky=downstream ! "
                "videoconvert ! autovideosink sync=false"
            )
            self.display_pipeline = Gst.parse_launch(display_str)
            self.appsrc = self.display_pipeline.get_by_name("src")
        
        elif self.mode == 'display':
            # Display only: webcam -> display
            capture_str = f"{base} ! autovideosink"
            self.capture_pipeline = Gst.parse_launch(capture_str)
        
        elif self.mode == 'record':
            # Recording only: webcam -> file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_file = f"recording_{timestamp}.mp4"
            capture_str = (
                f"{base} ! "
                "video/x-raw ! queue ! videoconvert ! video/x-raw,format=I420 ! "
                "x264enc tune=zerolatency bitrate=2000 speed-preset=fast ! "
                "video/x-h264,profile=baseline ! mp4mux ! "
                f"filesink location={self.recording_file}"
            )
            self.capture_pipeline = Gst.parse_launch(capture_str)
        
        elif self.mode == 'record-display':
            # Recording + Display: webcam -> tee -> file + display
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_file = f"recording_{timestamp}.mp4"
            capture_str = (
                f"{base} ! video/x-raw ! tee name=t "
                "t. ! queue ! videoconvert ! video/x-raw,format=I420 ! "
                "x264enc tune=zerolatency bitrate=2000 speed-preset=fast ! "
                "video/x-h264,profile=baseline ! mp4mux ! "
                f"filesink location={self.recording_file} "
                "t. ! queue ! videoconvert ! autovideosink"
            )
            self.capture_pipeline = Gst.parse_launch(capture_str)
        
        elif self.mode == 'record-detect-display':
            # Capture + recording pipeline: webcam -> tee -> file + appsink
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_file = f"recording_{timestamp}.mp4"
            capture_str = (
                f"{base} ! tee name=t "
                "t. ! queue max-size-buffers=10 ! videoconvert ! video/x-raw,format=I420 ! "
                "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
                "video/x-h264,profile=baseline ! mp4mux ! "
                f"filesink location={self.recording_file} "
                "t. ! queue max-size-buffers=2 leaky=downstream ! appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true"
            )
            self.capture_pipeline = Gst.parse_launch(capture_str)
            
            # Display pipeline: appsrc -> display
            display_str = (
                "appsrc name=src format=time is-live=true do-timestamp=true block=false max-bytes=0 "
                "! video/x-raw,format=BGR,width=640,height=480,framerate=30/1 "
                "! queue max-size-buffers=2 leaky=downstream ! "
                "videoconvert ! autovideosink sync=false"
            )
            self.display_pipeline = Gst.parse_launch(display_str)
            self.appsrc = self.display_pipeline.get_by_name("src")
        
        if self.verbose:
            print(f"\nPipeline mode: {self.mode}")
            if self.recording_file:
                print(f"Recording to: {self.recording_file}")
        
        print(f"Creating pipelines...")
        
        # Get appsink element (only if detection mode)
        if 'detect' in self.mode:
            appsink = self.capture_pipeline.get_by_name("sink")
            if appsink:
                appsink.connect("new-sample", self.on_new_sample)
            else:
                print("Warning: No appsink found in pipeline")
    
    def detect_objects_onnx(self, frame):
        """Run people + bottle ONNX model detection on a frame.

        Returns (detections_dict, boxes_list) where detections_dict counts classes
        and boxes_list contains box dicts with keys 'class','confidence','x','y','w','h'.
        """
        if not self.onnx_loaded:
            return defaultdict(int), []
        
        # Class names for the custom model
        class_names = ['yfood', 'pona', 'nocco']
        
        try:
            # Prepare input: model expects [batch, H, W, C] format in RGB
            # Resize to match model input (480, 640, 3)
            height, width = frame.shape[:2]
            input_frame = cv2.resize(frame, (640, 480))
            
            # Convert BGR to RGB (OpenCV loads as BGR, but model expects RGB)
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            input_frame = input_frame.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_frame, axis=0)  # Add batch dimension
            
            if self.verbose:
                print(f"  [ONNX] Input shape: {input_tensor.shape}")
                print(f"  [ONNX] Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            
            # Run inference
            outputs = self.onnx_session.run(
                [self.onnx_output_name],
                {self.onnx_input_name: input_tensor}
            )
            
            # Get output - shape varies by model
            detections = defaultdict(int)
            boxes = []
            output = outputs[0]  # Get first output
            
            # Remove batch dimension if present
            if len(output.shape) > 2 and output.shape[0] == 1:
                output = output[0]
            
            if self.verbose:
                print(f"  [ONNX] Output shape: {output.shape}")
                print(f"  [ONNX] Output range: [{output.min():.3f}, {output.max():.3f}]")
                unique_vals = np.unique(output)
                print(f"  [ONNX] Output unique values (first 20): {unique_vals[:20]}")
                print(f"  [ONNX] Output first row: {output[0] if len(output.shape) >= 1 else output}")
            
            # Handle different output formats
            if len(output.shape) == 2:
                # Format: [num_detections, 6] - each row is [x_center, y_center, width, height, confidence, class_id]
                num_detections, num_values = output.shape
                
                if self.verbose:
                    print(f"  [ONNX] Detection format: {num_detections} rows x {num_values} values")
                
                detection_count = 0
                confidence_threshold = 0.15
                
                # Process each detection (row)
                for i in range(num_detections):
                    detection = output[i]
                    
                    # Check if this is a valid detection (not padded zeros)
                    # Valid detections have reasonable x, y coordinates
                    if detection[0] == 0 and detection[1] == 0 and detection[2] == 0:
                        continue
                    
                    # Parse detection: [x_center, y_center, width, height, confidence, class_id]
                    x_center = float(detection[0])
                    y_center = float(detection[1])
                    w = float(detection[2])
                    h = float(detection[3])
                    confidence = float(detection[4]) if num_values >= 5 else 1.0
                    class_id = int(round(detection[5])) if num_values >= 6 else 0
                    
                    # Skip low confidence detections
                    if confidence < confidence_threshold:
                        continue
                    
                    # Model uses 1-indexed classes (1=yfood, 2=pona, 3=nocco), convert to 0-indexed
                    class_id = class_id - 1 if class_id > 0 else 0
                    
                    # Get class name
                    predicted_class = class_names[class_id] if class_id < len(class_names) else 'unknown'
                    
                    # Convert from center coordinates to corner coordinates
                    x = int(x_center - w / 2)
                    y = int(y_center - h / 2)
                    w = int(w)
                    h = int(h)
                    
                    # Clip to frame boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    detection_count += 1
                    
                    boxes.append({
                        'class': predicted_class,
                        'confidence': float(confidence),
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })
                    
                    detections[predicted_class] += 1
                    
                    if self.verbose:
                        print(f"  [ONNX] Detected {predicted_class} (class {class_id}): x={x}, y={y}, w={w}, h={h}, conf={confidence:.3f}")
                
                if self.verbose:
                    print(f"  [ONNX] Total detections: {detection_count}")
                    print(f"  [ONNX] Detections by class: {dict(detections)}")
                    
            elif len(output.shape) >= 3:
                # Format: [grid_h, grid_w, ...] - multi-value per cell
                grid_h, grid_w = output.shape[0], output.shape[1]
                cell_h = height / grid_h
                cell_w = width / grid_w
                
                if self.verbose:
                    print(f"  [ONNX] Grid size: {grid_h}x{grid_w}, Cell size: {cell_w:.1f}x{cell_h:.1f}")
                
                confidence_threshold = 0.15
                detection_count = 0
                max_conf = 0.0
                
                for i in range(grid_h):
                    for j in range(grid_w):
                        # Extract detection info
                        if len(output.shape) == 4:
                            det = output[i, j, 0, :]
                        else:
                            det = output[i, j, :]
                        
                        # Assuming output format: [objectness, class1_prob, class2_prob, class3_prob]
                        objectness = det[0]
                        
                        # Apply sigmoid if needed
                        if objectness < 0 or objectness > 1:
                            objectness = 1 / (1 + np.exp(-objectness))
                        
                        max_conf = max(max_conf, objectness)
                        
                        if objectness > confidence_threshold:
                            # Get class probabilities
                            class_probs = det[1:4] if len(det) >= 4 else [det[0], 0, 0]
                            class_probs = np.array([1 / (1 + np.exp(-p)) if (p < 0 or p > 1) else p for p in class_probs])
                            
                            # Get predicted class
                            class_idx = np.argmax(class_probs)
                            class_conf = class_probs[class_idx]
                            predicted_class = class_names[class_idx] if class_idx < len(class_names) else 'unknown'
                            
                            final_confidence = objectness * class_conf
                            detection_count += 1
                            
                            # Calculate bounding box
                            x = int(j * cell_w)
                            y = int(i * cell_h)
                            w = int(cell_w * 1.5)
                            h = int(cell_h * 1.5)
                            
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, width - x)
                            h = min(h, height - y)
                            
                            boxes.append({
                                'class': predicted_class,
                                'confidence': float(final_confidence),
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h
                            })
                            
                            detections[predicted_class] += 1
                            
                            if self.verbose:
                                print(f"  [ONNX] Detected {predicted_class} at grid ({i},{j}): objectness={objectness:.3f}, class_conf={class_conf:.3f}")
                
                if self.verbose:
                    print(f"  [ONNX] Max confidence found: {max_conf:.3f}, Threshold: {confidence_threshold}")
                    print(f"  [ONNX] Total detections: {detection_count}")
                    print(f"  [ONNX] Detections by class: {dict(detections)}")
            
            return detections, boxes
            
        except Exception as e:
            print(f"  [ONNX] Detection error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return defaultdict(int), []
    
    def detect_objects(self, frame):
        """Run object detection on frame (YOLO or both models)"""
        if not self.model_loaded:
            # Mock detection for demo
            import random
            mock_detections = {
                "person": random.randint(0, 3),
                "bottle": random.randint(0, 2),
                "cup": random.randint(0, 2),
            }
            return mock_detections, defaultdict(int), [], []
        
        yolo_detections = defaultdict(int)
        onnx_detections = defaultdict(int)
        yolo_boxes = []
        onnx_boxes = []
        
        # Run YOLO detection (if enabled)
        if self.yolo_loaded and (self.model_type == 'yolo' or self.compare_models):
            yolo_detections, yolo_boxes = self._detect_yolo(frame)

        # Run ONNX detection (if enabled)
        if self.onnx_loaded and (self.model_type == 'onnx' or self.compare_models or self.yolo_loaded is False):
            onnx_detections, onnx_boxes = self.detect_objects_onnx(frame)

        # Always return (yolo_detections, onnx_detections, yolo_boxes, onnx_boxes)
        return yolo_detections, onnx_detections, yolo_boxes, onnx_boxes
    
    def _detect_yolo(self, frame):
        """Run YOLO detection on frame"""
        if not self.yolo_loaded:
            return defaultdict(int), []
        
        # Real YOLO detection
        height, width = frame.shape[:2]
        
        # Create blob and forward pass
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # Process detections
        detections = defaultdict(int)
        confidence_threshold = 0.25  # Lowered from 0.5 for better detection
        nms_threshold = 0.4
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        if self.verbose and len(boxes) > 0:
            print(f"  [DEBUG] Pre-NMS detections: {len(boxes)}")
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        detection_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                class_name = self.classes[class_ids[i]]
                detections[class_name] += 1
                
                # Store box info for drawing
                detection_boxes.append({
                    'class': class_name,
                    'confidence': confidences[i],
                    'x': boxes[i][0],
                    'y': boxes[i][1],
                    'w': boxes[i][2],
                    'h': boxes[i][3]
                })
                
                if self.verbose:
                    print(f"  [DEBUG] Detected: {class_name} (confidence: {confidences[i]:.2f})")
            self.total_detections_ever += len(indices)
        
        return detections, detection_boxes
    
    def on_new_sample(self, appsink):
        """Callback for new frame from GStreamer pipeline"""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        
        # Get buffer and convert to numpy array
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Extract frame info
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        
        # Get buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        # Convert to numpy array (make writable copy for drawing)
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        ).copy()
        
        buffer.unmap(map_info)
        
        # Run detection every 10 frames for better FPS
        self.frame_count += 1
        
        # Always push frame to display (annotated or raw)
        display_frame = frame
        
        if self.frame_count % 10 == 0:
            # Run detection
            yolo_detections, onnx_detections, yolo_boxes, onnx_boxes = self.detect_objects(frame)
            self.object_counts = yolo_detections
            self.custom_counts = onnx_detections
            self.current_yolo_boxes = yolo_boxes
            self.current_onnx_boxes = onnx_boxes
            
            # Draw bounding boxes on the frame
            self.draw_boxes(frame)
            self.last_annotated_frame = frame.copy()
            display_frame = frame
        elif self.last_annotated_frame is not None:
            # Reuse last annotated frame between detections
            display_frame = self.last_annotated_frame
        
        # Push to display pipeline if available
        if self.appsrc:
            # Create GStreamer buffer from numpy array
            data = display_frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            
            # Push buffer to appsrc
            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK and self.verbose:
                print(f"  [DEBUG] Failed to push buffer: {ret}")
            
        # Save frame every 30 frames for debugging (only on detection frames)
        if self.verbose and self.frame_count % 30 == 0 and self.frame_count % 10 == 0:
            cv2.imwrite(f"debug_frame_{self.frame_count}.jpg", display_frame)
            print(f"  [DEBUG] Saved debug_frame_{self.frame_count}.jpg")
        
        # Print results every 2 seconds (only on detection frames)
        if self.frame_count % 10 == 0:
            current_time = time.time()
            if current_time - self.last_print_time >= 2.0:
                self.print_detections()
                self.last_print_time = current_time
        
        return Gst.FlowReturn.OK
    
    def print_detections(self):
        """Print detection results to terminal"""
        print("\n" + "="*70)
        print(f"Frame #{self.frame_count} - Detection Results")
        print("="*70)
        
        if self.compare_models:
            # Show comparison between models
            print("\nYOLO Model:")
            if not self.object_counts:
                print("  No objects detected")
            else:
                yolo_total = sum(self.object_counts.values())
                print(f"  Total: {yolo_total} objects")
                for obj_class, count in sorted(self.object_counts.items()):
                    emoji = self.get_emoji(obj_class)
                    print(f"    {emoji} {obj_class:15s}: {count}")
            
            print("\nCustom ONNX Model (Bottle Detection):")
            if not self.custom_counts:
                print("  No bottles detected")
            else:
                onnx_total = sum(self.custom_counts.values())
                print(f"  Total: {onnx_total} bottles")
                for obj_class, count in sorted(self.custom_counts.items()):
                    emoji = self.get_emoji(obj_class)
                    print(f"    {emoji} {obj_class:15s}: {count}")
        else:
            # Show single model results
            model_name = "Custom ONNX" if self.model_type == 'onnx' else "YOLO"
            counts = self.custom_counts if self.model_type == 'onnx' else self.object_counts
            
            print(f"\nModel: {model_name}")
            if not counts:
                print("No objects detected")
            else:
                total = sum(counts.values())
                print(f"Total objects: {total}")
                print("\nBreakdown:")
                for obj_class, count in sorted(counts.items()):
                    emoji = self.get_emoji(obj_class)
                    print(f"  {emoji} {obj_class:15s}: {count}")
        
        print("="*70)
    
    def get_emoji(self, class_name):
        """Get emoji for object class"""
        emoji_map = {
            "person": "👤",
            "bottle": "🍾",
            "cup": "☕",
            "laptop": "💻",
            "cell phone": "📱",
            "book": "📚",
            "chair": "🪑",
            "keyboard": "⌨️",
            "mouse": "🖱️",
        }
        return emoji_map.get(class_name, "📦")
    
    def process_image(self, image_path):
        """Process a single image file with object detection"""
        import os
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Error: Failed to read image: {image_path}")
            return
        
        print(f"\n📷 Processing image: {image_path}")
        print(f"   Size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Resize if needed
        height, width = frame.shape[:2]
        if width > 1920 or height > 1080:
            scale = min(1920/width, 1080/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            print(f"   Resized to: {new_width}x{new_height}")
        
        # Run detection
        print("\n🔍 Running detection...")
        primary_detections, secondary_detections, primary_boxes, secondary_boxes = self.detect_objects(frame)
        
        # Assign based on model type
        if self.compare_models:
            self.object_counts = primary_detections  # YOLO
            self.custom_counts = secondary_detections  # ONNX
            self.current_yolo_boxes = primary_boxes
            self.current_onnx_boxes = secondary_boxes
        elif self.model_type == 'onnx':
            # When using ONNX model, the ONNX results come in secondary (index 1, 3)
            self.custom_counts = secondary_detections  # ONNX is in secondary
            self.object_counts = primary_detections  # YOLO is in primary (empty)
            self.current_onnx_boxes = secondary_boxes  # ONNX boxes in secondary
            self.current_yolo_boxes = primary_boxes  # YOLO boxes in primary (empty)
        else:
            self.object_counts = primary_detections  # YOLO is primary
            self.custom_counts = secondary_detections  # Empty
            self.current_yolo_boxes = primary_boxes
            self.current_onnx_boxes = secondary_boxes
        
        # Draw bounding boxes
        self.draw_boxes(frame)
        
        # Print results
        print("\n" + "="*70)
        print("📋 Detection Results")
        print("="*70)
        
        if self.compare_models:
            print("\n🔵 YOLO Model:")
            if not self.object_counts:
                print("  No objects detected")
            else:
                yolo_total = sum(self.object_counts.values())
                print(f"  Total: {yolo_total} objects")
                for obj_class, count in sorted(self.object_counts.items()):
                    emoji = self.get_emoji(obj_class)
                    print(f"    {emoji} {obj_class:15s}: {count}")
            
            print("\n🟢 Custom ONNX Model:")
            if not self.custom_counts:
                print("  No objects detected")
            else:
                onnx_total = sum(self.custom_counts.values())
                print(f"  Total: {onnx_total} objects")
                for obj_class, count in sorted(self.custom_counts.items()):
                    emoji = self.get_emoji(obj_class)
                    print(f"    {emoji} {obj_class:15s}: {count}")
        else:
            model_name = "Custom ONNX" if self.model_type == 'onnx' else "YOLO"
            counts = self.custom_counts if self.model_type == 'onnx' else self.object_counts
            
            print(f"\nModel: {model_name}")
            if not counts:
                print("No objects detected")
            else:
                total = sum(counts.values())
                print(f"Total objects: {total}")
                print("\nBreakdown:")
                for obj_class, count in sorted(counts.items()):
                    emoji = self.get_emoji(obj_class)
                    print(f"  {emoji} {obj_class:15s}: {count}")
        
        print("="*70)
        
        # Save output image
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{basename}_detected.jpg"
        cv2.imwrite(output_path, frame)
        print(f"\n✓ Annotated image saved: {output_path}")
        
        # Display if in display mode
        if 'display' in self.mode:
            print("\n👁️  Press any key to close the window...")
            cv2.imshow('Detection Results', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def draw_boxes(self, frame):
        """Draw bounding boxes with labels on frame"""
        if self.compare_models:
            # Draw YOLO boxes in blue
            for box in self.current_yolo_boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                label = f"YOLO: {box['class']} {box['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - label_size[1] - 4), (x + label_size[0], y), (255, 0, 0), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw ONNX boxes in green
            for box in self.current_onnx_boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                label = f"ONNX: {box['class']} {box['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - label_size[1] - 4), (x + label_size[0], y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            # Draw boxes for both models when not in compare mode so people (YOLO) and bottles (ONNX)
            # are both visible on the annotated output.
            # YOLO boxes (if present) in cyan
            for box in self.current_yolo_boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                label = f"{box.get('class','obj')} {box.get('confidence',0):.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - label_size[1] - 4), (x + label_size[0], y), (255, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # ONNX boxes (if present) in yellow-green
            for box in self.current_onnx_boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                label = f"{box.get('class','obj')} {box.get('confidence',0):.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - label_size[1] - 4), (x + label_size[0], y), (0, 255, 255), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def annotate_frame(self, frame):
        """Run detection on a frame, draw boxes, and return (annotated_frame, counts).

        This provides a simple compatible API for `CameraService` which expects
        an (annotated_frame, counts) tuple.
        """
        # Run detection
        try:
            yolo_detections, onnx_detections, yolo_boxes, onnx_boxes = self.detect_objects(frame)
        except Exception as e:
            # If detection fails, return the original frame and empty counts
            if self.verbose:
                print(f"[annotate_frame] detection error: {e}")
            return frame.copy(), defaultdict(int)
        # We now return a consistent tuple: (yolo_detections, onnx_detections, yolo_boxes, onnx_boxes)
        # Merge counts (give priority to summed counts)

        merged = defaultdict(int)
        for k, v in yolo_detections.items():
            merged[k] += int(v or 0)
        for k, v in onnx_detections.items():
            merged[k] += int(v or 0)

        # Update current boxes for drawing
        self.current_yolo_boxes = yolo_boxes
        self.current_onnx_boxes = onnx_boxes

        # no placeholder generation here; rely on real model boxes if available

        # Draw boxes on a copy of the frame
        annotated = frame.copy()
        try:
            self.draw_boxes(annotated)
            self.last_annotated_frame = annotated.copy()
        except Exception as e:
            if self.verbose:
                print(f"[annotate_frame] draw_boxes error: {e}")

        return annotated, merged
    
    def on_message(self, bus, message):
        """Handle GStreamer bus messages"""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ Error: {err}")
            print(f"Debug: {debug}")
            self.loop.quit()
        elif msg_type == Gst.MessageType.EOS:
            print("End of stream")
            self.loop.quit()
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.capture_pipeline or (self.display_pipeline and message.src == self.display_pipeline):
                old_state, new_state, pending = message.parse_state_changed()
                pipeline_name = "Display" if message.src == self.display_pipeline else "Capture"
                if self.verbose:
                    print(f"{pipeline_name} pipeline: {old_state.value_nick} -> {new_state.value_nick}")
        
        return True
    
    def run(self):
        """Start the pipelines and main loop"""
        # Check if input is an image file
        if self.input_source and self.input_source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            self.is_image_input = True
            self.process_image(self.input_source)
            return
        
        # Create pipelines for video/camera input
        self.create_pipelines()
        
        # Set up bus for capture pipeline
        capture_bus = self.capture_pipeline.get_bus()
        capture_bus.add_signal_watch()
        capture_bus.connect("message", self.on_message)
        
        # Set up bus for display pipeline if it exists
        if self.display_pipeline:
            display_bus = self.display_pipeline.get_bus()
            display_bus.add_signal_watch()
            display_bus.connect("message", self.on_message)
        
        # Start pipelines
        mode_desc = {
            'detect': 'Object Detection',
            'detect-display': 'Object Detection + Display',
            'display': 'Display Only',
            'record': 'Recording',
            'record-display': 'Recording + Display',
            'record-detect-display': 'Recording + Detection + Display'
        }
        print(f"\nStarting pipeline: {mode_desc.get(self.mode, self.mode)}")
        if self.recording_file:
            print(f"   Recording to: {self.recording_file}")
        print("Press Ctrl+C to stop\n")
        
        # Start display pipeline first if it exists
        if self.display_pipeline:
            ret = self.display_pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("❌ Failed to start display pipeline")
                return
        
        # Start capture pipeline
        ret = self.capture_pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Failed to start capture pipeline")
            # Get more detailed error info
            bus = self.capture_pipeline.get_bus()
            msg = bus.timed_pop_filtered(1000000000, Gst.MessageType.ERROR)  # 1 second timeout
            if msg:
                err, debug = msg.parse_error()
                print(f"   Error: {err}")
                if self.verbose and debug:
                    print(f"   Debug: {debug}")
            return
        
        # Run main loop
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping pipelines...")
        finally:
            # Send EOS to properly finalize recording
            if self.recording_file:
                print("Finalizing recording...")
                self.capture_pipeline.send_event(Gst.Event.new_eos())
                # Wait for EOS to be processed
                bus = self.capture_pipeline.get_bus()
                bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            
            # Stop pipelines
            self.capture_pipeline.set_state(Gst.State.NULL)
            if self.display_pipeline:
                self.display_pipeline.set_state(Gst.State.NULL)
            
            if self.recording_file:
                print(f"✓ Recording saved: {self.recording_file}")
            else:
                print("✓ Pipelines stopped")


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='GStreamer Object Detection Pipeline')
    parser.add_argument('--input', type=str, help='Input source: image file (.jpg, .png) or video file (.mp4, .avi). If not specified, uses camera.')
    parser.add_argument('--device', type=int, default=0, help='Video device number (default: 0, ignored if --input is specified)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output')
    parser.add_argument('--model', type=str, choices=['yolo', 'onnx'], default='yolo',
                       help='Model to use: yolo or onnx (default: yolo)')
    parser.add_argument('--custom-model', type=str, dest='custom_model', metavar='FILE',
                       help='Path to custom model file (.onnx). Auto-detects format.')
    parser.add_argument('--compare', action='store_true', 
                       help='Run both models and compare results')
    parser.add_argument('--mode', type=str, 
                       choices=['detect', 'detect-display', 'display', 'record', 'record-display', 'record-detect-display'],
                       default='detect',
                       help='Pipeline mode: detect (default), detect-display, display, record, record-display, record-detect-display')
    args = parser.parse_args()
    
    # Validate custom model file if provided
    if args.custom_model:
        if not os.path.exists(args.custom_model):
            print(f"❌ Error: Model file not found: {args.custom_model}")
            return
        if not args.custom_model.endswith('.onnx'):
            print(f"⚠ Warning: File does not have .onnx extension: {args.custom_model}")
            print("  Attempting to load anyway...")
    
    print("="*70)
    print("  GStreamer Object Detection Pipeline")
    print("="*70)
    
    if args.input:
        print(f"Input source: {args.input}")
    else:
        print(f"Video device: /dev/video{args.device}")
    
    print(f"Pipeline mode: {args.mode}")
    
    # Only show model info if detection is involved
    if 'detect' in args.mode:
        if args.compare:
            print("Detection: Comparison (YOLO + Custom ONNX)")
        else:
            model_name = args.model.upper()
            if args.custom_model:
                model_name = f"Custom ({os.path.basename(args.custom_model)})"
            print(f"Detection model: {model_name}")
    
    if args.verbose:
        print("Verbose: ON")
    print()
    
    detector = ObjectDetector(
        video_device=args.device, 
        verbose=args.verbose,
        model_type=args.model,
        compare_models=args.compare,
        custom_model_file=args.custom_model,
        mode=args.mode,
        input_source=args.input
    )
    detector.run()


if __name__ == "__main__":
    main()
