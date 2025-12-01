# Object Detection with GStreamer

Multi-model object detection pipeline supporting YOLO and custom ONNX models.

## Features

- ✅ GStreamer-based video pipeline
- ✅ YOLO-tiny object detection (80 COCO classes)
- ✅ Custom ONNX model support (bottle detection)
- ✅ Model comparison mode
- ✅ Hardware adaptable (Pi 5, Pi Zero, ESP32)

## Usage

### Basic Detection (YOLO)
```bash
./run_detector.sh
```

### Custom ONNX Model (Default)
```bash
./run_detector.sh --model onnx
```

### Custom ONNX Model (Specify File)
```bash
./run_detector.sh --custom-model my_model.onnx
```

### Compare Both Models
```bash
./run_detector.sh --compare
```

### Options
```bash
./run_detector.sh [options]

Options:
  --device N            Video device number (default: 0)
  --model MODEL         Model to use: yolo or onnx (default: yolo)
  --custom-model FILE   Path to custom ONNX model file (auto-detects format)
  --compare             Run both models and compare results
  --verbose             Enable verbose debug output
```

## Examples

```bash
# Use default ONNX model with verbose output
./run_detector.sh --model onnx --verbose

# Use specific custom ONNX model
./run_detector.sh --custom-model bottle_detector_v2.onnx

# Compare YOLO vs custom model on camera 2
./run_detector.sh --compare --device 2 --custom-model my_model.onnx

# YOLO with debug frames
./run_detector.sh --model yolo --verbose

# Any .onnx file - format auto-detected
./run_detector.sh --custom-model /path/to/model.onnx
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
