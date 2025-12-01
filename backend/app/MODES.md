# GStreamer Object Detector - Mode Examples

## 🎯 All 6 Modes

### 1. Detection Only (default)
```bash
./run_detector.sh
./run_detector.sh --model onnx
```
**Use case:** Headless operation, logging detections

---

### 2. Detection + Display
```bash
./run_detector.sh --mode detect-display
./run_detector.sh --mode detect-display --model onnx
```
**Use case:** Real-time monitoring with visual feedback

---

### 3. Display Only
```bash
./run_detector.sh --mode display
./run_detector.sh --mode display --device 2
```
**Use case:** Camera preview, testing camera feeds

---

### 4. Recording Only
```bash
./run_detector.sh --mode record
```
**Use case:** Save footage for later analysis
**Output:** `recording_YYYYMMDD_HHMMSS.mp4`

---

### 5. Recording + Display
```bash
./run_detector.sh --mode record-display
./run_detector.sh --mode record-display --device 2
```
**Use case:** Record while monitoring live

---

### 6. Recording + Detection + Display (Full Featured)
```bash
./run_detector.sh --mode record-detect-display --model onnx
./run_detector.sh --mode record-detect-display --compare --verbose
```
**Use case:** Complete monitoring solution with logging

---

## 🔍 Mode Comparison

| Mode | Detection | Display | Recording | Use Case |
|------|-----------|---------|-----------|----------|
| detect | ✅ | ❌ | ❌ | Headless monitoring |
| detect-display | ✅ | ✅ | ❌ | Live monitoring |
| display | ❌ | ✅ | ❌ | Camera preview |
| record | ❌ | ❌ | ✅ | Silent recording |
| record-display | ❌ | ✅ | ✅ | Record with preview |
| record-detect-display | ✅ | ✅ | ✅ | Full featured |

---

## 📝 Tips

- **Recording format:** H.264/MP4, 2000 kbps bitrate, baseline profile
- **Recording location:** Same directory as script (`recording_YYYYMMDD_HHMMSS.mp4`)
- **Stopping recording:** Press Ctrl+C - file will be properly finalized
- **Display requires:** X11 or Wayland display server
- **Detection works:** With any mode containing "detect"

## ⚠️ Important

**Always stop recording with Ctrl+C** to ensure the MP4 file is properly finalized with the moov atom. Using `timeout` or `kill -9` will create invalid files.
