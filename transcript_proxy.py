"""
LearnEngine - Local Server
Serves the app + fetches YouTube transcripts on your Mac.
Start: python3 transcript_proxy.py
Open:  http://localhost:8219
Stop:  Ctrl+C
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.parse

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    print("Installing youtube-transcript-api...")
    import subprocess
    subprocess.check_call(["pip3", "install", "youtube-transcript-api"])
    from youtube_transcript_api import YouTubeTranscriptApi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        # Transcript API
        if parsed.path == "/transcript":
            params = urllib.parse.parse_qs(parsed.query)
            video_id = (params.get("v") or [None])[0]
            if not video_id:
                return self._json(400, {"error": "Use /transcript?v=VIDEO_ID"})
            try:
                ytt = YouTubeTranscriptApi()
                fetched = ytt.fetch(video_id, languages=["en"])
                text = " ".join([s.text for s in fetched])
                return self._json(200, {
                    "video_id": video_id,
                    "full_text": text,
                    "word_count": len(text.split()),
                })
            except Exception as e:
                return self._json(400, {"error": str(e)})

        # Serve index.html for root
        if parsed.path in ("/", "/index.html"):
            filepath = os.path.join(SCRIPT_DIR, "index.html")
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return

        # 404
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def _json(self, code, data):
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        print(f"  [{args[1]}] {args[0]}")


if __name__ == "__main__":
    port = 8219
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"""
  ╔═══════════════════════════════════════════╗
  ║         LearnEngine is running!           ║
  ╠═══════════════════════════════════════════╣
  ║  Computer: http://localhost:{port}          ║
  ║  Phone:    http://<your-mac-ip>:{port}      ║
  ║  Press Ctrl+C to stop                     ║
  ╚═══════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.server_close()
