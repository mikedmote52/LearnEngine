"""
LearnEngine Transcript Proxy - runs on your Mac to fetch YouTube transcripts.
Start this with: python3 transcript_proxy.py
Then LearnEngine will auto-fetch transcripts from any YouTube video.
Press Ctrl+C to stop.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import re
import urllib.parse

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    print("Installing youtube-transcript-api...")
    import subprocess
    subprocess.check_call(["pip3", "install", "youtube-transcript-api"])
    from youtube_transcript_api import YouTubeTranscriptApi


class TranscriptHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        # GET /transcript?v=VIDEO_ID
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        video_id = (params.get("v") or [None])[0]

        if not video_id or parsed.path != "/transcript":
            self.send_response(404)
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Use /transcript?v=VIDEO_ID"}).encode())
            return

        try:
            ytt = YouTubeTranscriptApi()
            fetched = ytt.fetch(video_id, languages=["en"])
            text = " ".join([s.text for s in fetched])
            word_count = len(text.split())
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "video_id": video_id,
                "full_text": text,
                "word_count": word_count,
            }).encode())
        except Exception as e:
            self.send_response(400)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        print(f"  [{args[1]}] {args[0]}")


if __name__ == "__main__":
    port = 8219
    server = HTTPServer(("127.0.0.1", port), TranscriptHandler)
    print(f"\n  LearnEngine Transcript Proxy running on http://localhost:{port}")
    print(f"  Test: http://localhost:{port}/transcript?v=UF8uR6Z6KLc")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.server_close()
