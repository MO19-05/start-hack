# Object Detection with GStreamer

Multi-model object detection pipeline supporting YOLO and custom ONNX models.

## Features

- ✅ GStreamer-based video pipeline
- ✅ YOLO-tiny object detection (80 COCO classes)
- ✅ Custom ONNX model support (bottle detection)
- ✅ Model comparison mode
- ✅ Hardware adaptable (Pi 5, Pi Zero, ESP32)

## Usage

### Pipeline Modes

The detector supports 6 different modes:

1. **`detect`** - Object detection only (default)
2. **`detect-display`** - Detection + live video display
3. **`display`** - Just display video stream
4. **`record`** - Record video to file
5. **`record-display`** - Record + display
6. **`record-detect-display`** - Record + detect + display (all features)

### Basic Usage

```bash
# Detection only (default)
./run_detector.sh

# Detection with live display
./run_detector.sh --mode detect-display

# Just display video
./run_detector.sh --mode display

# Record video
./run_detector.sh --mode record

# Record with display
./run_detector.sh --mode record-display

# Full featured: record + detect + display
./run_detector.sh --mode record-detect-display
```

### Custom ONNX Model

```bash
# Use custom model with detection
./run_detector.sh --custom-model my_model.onnx

# Custom model with display
./run_detector.sh --custom-model my_model.onnx --mode detect-display

# Record with custom model detection
./run_detector.sh --custom-model my_model.onnx --mode record-detect-display
```

### Compare Both Models
```bash
# Compare YOLO vs ONNX with display
./run_detector.sh --compare --mode detect-display
```

### Options
```bash
./run_detector.sh [options]

Options:
  --device N            Video device number (default: 0)
  --mode MODE           Pipeline mode (default: detect)
                        Choices: detect, detect-display, display,
                                record, record-display, record-detect-display
  --model MODEL         Model to use: yolo or onnx (default: yolo)
  --custom-model FILE   Path to custom ONNX model file (auto-detects format)
  --compare             Run both models and compare results
  --verbose             Enable verbose debug output
```

## Examples

### Detection Examples
```bash
# YOLO detection with verbose output
./run_detector.sh --model yolo --verbose

# Custom ONNX model detection
./run_detector.sh --custom-model bottle_detector_v2.onnx

# Compare both models with display
./run_detector.sh --compare --mode detect-display
```

### Display Examples
```bash
# Just view camera feed
./run_detector.sh --mode display

# View camera with detection overlay
./run_detector.sh --mode detect-display --model onnx

# Use different camera with display
./run_detector.sh --mode display --device 2
```

### Recording Examples
```bash
# Record video only (press Ctrl+C to stop)
./run_detector.sh --mode record

# Record with live preview
./run_detector.sh --mode record-display

# Record with detection and display
./run_detector.sh --mode record-detect-display --model onnx

# Record with custom model on camera 2
./run_detector.sh --mode record-detect-display --device 2 --custom-model my_model.onnx
```

**Recording Details:**
- Format: H.264/MP4 (baseline profile)
- Bitrate: 2000 kbps
- Output: `recording_YYYYMMDD_HHMMSS.mp4` in current directory
- **Important:** Always stop with Ctrl+C to finalize the file properly

**Playing Recordings:**
```bash
gst-play-1.0 recording_20251129_231938.mp4
vlc recording_20251129_231938.mp4
ffplay recording_20251129_231938.mp4
```

### Advanced Examples
```bash
# Full featured with comparison mode
./run_detector.sh --mode record-detect-display --compare --verbose

# Record from specific camera with custom model
./run_detector.sh --mode record-display --device 2 --custom-model /path/to/model.onnx
```

## Models

### YOLO-tiny
- **Classes**: 80 COCO objects (person, bottle, cup, laptop, etc.)
- **Performance**: 30 FPS on Pi 5, 3-5 FPS on Pi Zero
- **Files**: `yolov3-tiny.weights`, `yolov3-tiny.cfg`, `coco.names`

### Custom ONNX (Bottle Detection)
- **Classes**: Bottle detection
- **Input**: 480x640x3 (HWC format)
- **Output**: 15x20 grid with confidence scores
- **File**: `bottle_classification_2025-11-29_20-35-40.onnx`

## Setup

```bash
# Initial setup
./setup_detector.sh

# This installs:
# - GStreamer libraries
# - Python dependencies (OpenCV, ONNX Runtime)
# - Model files (YOLO weights)
```

## Output Format

### Single Model Mode
```
📹 Frame #45 - Detection Results
======================================================================
Model: Custom ONNX
Total objects: 2

Breakdown:
  🍾 bottle         : 2
======================================================================
```

### Comparison Mode
```
📹 Frame #45 - Detection Results
======================================================================
🔵 YOLO Model:
  Total: 3 objects
    👤 person         : 1
    💻 laptop         : 1
    🍾 bottle         : 1

🟢 Custom ONNX Model (Bottle Detection):
  Total: 2 bottles
    🍾 bottle         : 2
======================================================================
```

## Adding Custom ONNX Models

**Easy way (just specify filename):**
```bash
./run_detector.sh --custom-model your_model.onnx
```

The script automatically:
- ✅ Detects ONNX format from `.onnx` extension
- ✅ Loads the model
- ✅ Validates file exists

**For different model architectures:**
1. Place your `.onnx` file anywhere accessible
2. Run with `--custom-model /path/to/model.onnx`
3. If input/output format differs, adjust preprocessing in `object_detector.py`:
   - Input preprocessing: lines 105-110
   - Output parsing: lines 119-135

## Video Devices

Check available cameras:
```bash
ls /dev/video*
```

Test specific camera:
```bash
./run_detector.sh --device 2
```

