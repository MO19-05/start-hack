#!/usr/bin/env bash
# Setup script for object detection pipeline

echo "=========================================="
echo "  Object Detection Setup"
echo "=========================================="

# Install system dependencies
echo ""
echo "📦 Installing GStreamer and dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    v4l-utils \
    python3-venv \
    python3-dev

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Download YOLO model files (optional - script works without them)
echo ""
echo "📥 Downloading YOLO model files (optional)..."
if [ ! -f "yolov3-tiny.weights" ]; then
    wget https://pjreddie.com/media/files/yolov3-tiny.weights
fi

if [ ! -f "yolov3-tiny.cfg" ]; then
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
fi

if [ ! -f "coco.names" ]; then
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the detector:"
echo "  source venv/bin/activate"
echo "  python3 object_detector.py"
echo ""
echo "Or use the run script:"
echo "  ./run_detector.sh"
echo ""
echo "Note: If model files are missing, the script will run in demo mode"

