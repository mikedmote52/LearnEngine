#!/bin/bash
# LearnEngine - Quick Start
# Run this script to install dependencies and launch the app.

cd "$(dirname "$0")"

echo ""
echo "  ╔═══════════════════════════════╗"
echo "  ║       LearnEngine v1.0        ║"
echo "  ╚═══════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: Python 3 is required but not installed."
    echo "  Install it from https://www.python.org/downloads/"
    exit 1
fi

# Install dependencies
echo "  Installing dependencies..."
pip3 install -r requirements.txt --quiet --break-system-packages 2>/dev/null || \
pip3 install -r requirements.txt --quiet

# Create data directory
mkdir -p data

echo ""
echo "  Starting LearnEngine..."
echo "  Open http://127.0.0.1:5050 in your browser"
echo "  Press Ctrl+C to stop"
echo ""
echo "  Set LEARNENGINE_HOST=0.0.0.0 to allow network access"
echo "  Set ANTHROPIC_API_KEY to skip manual key setup"
echo ""

python3 app.py
