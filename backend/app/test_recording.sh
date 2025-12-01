#!/bin/bash
# Test recording functionality

cd /home/gjk/projects/start-hack-one-ware/src
source venv/bin/activate

echo "=========================================="
echo "  Testing Recording Modes"
echo "=========================================="
echo ""
echo "Recording 5-second test videos..."
echo ""

# Test record mode
echo "1️⃣  Testing: record mode"
(sleep 5 && pkill -INT python3) &
python3 object_detector.py --mode record > /dev/null 2>&1
wait
LAST_FILE=$(ls -t recording_*.mp4 2>/dev/null | head -1)
if [ -f "$LAST_FILE" ]; then
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$LAST_FILE" 2>/dev/null)
    echo "   ✓ Created: $LAST_FILE (${DURATION}s)"
else
    echo "   ✗ Failed"
fi
echo ""

# Test record-display mode  
echo "2️⃣  Testing: record-display mode"
(sleep 5 && pkill -INT python3) &
python3 object_detector.py --mode record-display > /dev/null 2>&1
wait
LAST_FILE=$(ls -t recording_*.mp4 2>/dev/null | head -1)
if [ -f "$LAST_FILE" ]; then
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$LAST_FILE" 2>/dev/null)
    echo "   ✓ Created: $LAST_FILE (${DURATION}s)"
else
    echo "   ✗ Failed"
fi
echo ""

# Test record-detect-display mode
echo "3️⃣  Testing: record-detect-display mode"
(sleep 5 && pkill -INT python3) &
python3 object_detector.py --mode record-detect-display --model onnx > /dev/null 2>&1
wait
LAST_FILE=$(ls -t recording_*.mp4 2>/dev/null | head -1)
if [ -f "$LAST_FILE" ]; then
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$LAST_FILE" 2>/dev/null)
    echo "   ✓ Created: $LAST_FILE (${DURATION}s)"
else
    echo "   ✗ Failed"
fi
echo ""

echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
TOTAL=$(ls recording_*.mp4 2>/dev/null | wc -l)
echo "Total recordings: $TOTAL"
echo ""
echo "To play a recording:"
echo "  gst-play-1.0 recording_YYYYMMDD_HHMMSS.mp4"
echo "  vlc recording_YYYYMMDD_HHMMSS.mp4"
echo "  ffplay recording_YYYYMMDD_HHMMSS.mp4"
echo ""

