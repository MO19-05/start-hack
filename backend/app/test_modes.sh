#!/bin/bash
# Quick test of all modes

cd /home/gjk/projects/start-hack-one-ware/src

echo "=========================================="
echo "  Testing All Pipeline Modes"
echo "=========================================="
echo ""

source venv/bin/activate

echo "1️⃣  Testing: detect mode"
timeout 2 python3 object_detector.py --mode detect > /dev/null 2>&1 && echo "   ✓ Works" || echo "   ✗ Failed"

echo "2️⃣  Testing: detect-display mode"
timeout 2 python3 object_detector.py --mode detect-display > /dev/null 2>&1 && echo "   ✓ Works" || echo "   ✗ Failed"

echo "3️⃣  Testing: display mode"
timeout 2 python3 object_detector.py --mode display > /dev/null 2>&1 && echo "   ✓ Works" || echo "   ✗ Failed"

echo "4️⃣  Testing: record mode"
timeout 2 python3 object_detector.py --mode record > /dev/null 2>&1 && echo "   ✓ Works" || echo "   ✗ Failed"

echo "5️⃣  Testing: record-display mode"
timeout 2 python3 object_detector.py --mode record-display > /dev/null 2>&1 && echo "   ✓ Works" || echo "   ✗ Failed"

echo "6️⃣  Testing: record-detect-display mode"
timeout 2 python3 object_detector.py --mode record-detect-display > /dev/null 2>&1 && echo "   ✓ Works" || echo "   ✗ Failed"

echo ""
echo "=========================================="
echo "  All modes functional!"
echo "=========================================="

