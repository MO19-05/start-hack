#!/bin/bash
# Run object detector with virtual environment

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./setup_detector.sh"
    exit 1
fi

# Activate virtual environment and run detector
source venv/bin/activate
python3 object_detector.py "$@"
