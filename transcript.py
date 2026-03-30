"""
YouTube Transcript Fetcher
Extracts transcripts from YouTube videos using youtube-transcript-api v1.x.
Falls back to manual paste if API fails (age-restricted, disabled captions, etc.)
"""

import re
import json
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',  # bare video ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def fetch_transcript(url: str, language: str = "en") -> dict:
    """
    Fetch transcript for a YouTube video.
    Returns a dict with video_id, full_text, segments (timestamped), and metadata.
    Compatible with youtube-transcript-api v1.x (instance-based API).
    """
    video_id = extract_video_id(url)

    try:
        ytt = YouTubeTranscriptApi()

        # Fetch transcript - tries requested language first
        fetched = ytt.fetch(video_id, languages=[language])

        # Build segments list and full text
        segments = []
        text_parts = []
        for snippet in fetched:
            segments.append({
                "start": snippet.start,
                "duration": snippet.duration,
                "text": snippet.text,
            })
            text_parts.append(snippet.text)

        full_text = " ".join(text_parts)

        # Calculate duration from last segment
        duration_seconds = 0
        if segments:
            last = segments[-1]
            duration_seconds = last["start"] + last["duration"]

        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "full_text": full_text,
            "segments": segments,
            "language": language,
            "is_generated": False,
            "duration_seconds": duration_seconds,
            "fetched_at": datetime.now().isoformat(),
            "word_count": len(full_text.split()),
        }

    except Exception as e:
        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "error": str(e),
            "full_text": None,
            "segments": [],
            "fetched_at": datetime.now().isoformat(),
        }


def get_video_metadata(url: str) -> dict:
    """Get basic video metadata from the video ID (title requires separate fetch)."""
    video_id = extract_video_id(url)
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = fetch_transcript(sys.argv[1])
        if result.get("error"):
            print(f"Error: {result['error']}")
            print("You can paste the transcript manually in the web interface.")
        else:
            print(f"Fetched {result['word_count']} words in {result['language']}")
            print(f"Duration: {result['duration_seconds']:.0f}s")
            print(f"\nFirst 500 chars:\n{result['full_text'][:500]}")
    else:
        print("Usage: python transcript.py <youtube_url>")
