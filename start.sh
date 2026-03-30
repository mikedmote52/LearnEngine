#!/bin/bash
# LearnEngine - Start the local server
# Usage: ./start.sh
cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 not found. Install Python 3 first."
  exit 1
fi

# Check/install youtube-transcript-api
python3 -c "import youtube_transcript_api" 2>/dev/null || {
  echo "Installing youtube-transcript-api..."
  pip3 install youtube-transcript-api
}

# Get local IP for phone access
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')

echo ""
echo "  Starting LearnEngine..."
echo "  Computer: http://localhost:8219"
[ -n "$LOCAL_IP" ] && echo "  Phone:    http://$LOCAL_IP:8219"
echo ""

python3 transcript_proxy.py
