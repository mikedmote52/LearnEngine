"""
LearnEngine - Vercel Serverless API (proxy)

Holds the OpenRouter API key server-side. The browser client never sees it.

Model routing:
  Bulk formatting (distractors, scoring, summaries, review packs)
    -> anthropic/claude-haiku-4.5
  Hard reasoning (deep analysis, synthesis, teaching follow-ups)
    -> anthropic/claude-sonnet-4.5
  Embeddings
    -> openai/text-embedding-3-small

Optimizations:
  - Prompt caching on transcripts (OpenRouter cache_control)
  - Batched generation (N questions in one call)
  - Token caps per call site
  - CORS locked to GH Pages + localhost dev origins

Stateless: all learner state lives client-side in IndexedDB.
"""

import json
import os
import re
import base64
import time
import tempfile
import concurrent.futures
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, make_response, Response
import urllib.request
import urllib.error
import urllib.parse

app = Flask(__name__)

# ---- Models (OpenRouter naming) ----
HAIKU = os.environ.get("LEARNENGINE_MODEL_HAIKU", "anthropic/claude-haiku-4.5")
SONNET = os.environ.get("LEARNENGINE_MODEL_SONNET", "anthropic/claude-sonnet-4.5")
EMBED_MODEL = os.environ.get("LEARNENGINE_EMBED_MODEL", "openai/text-embedding-3-small")
NEVER_USE_OPUS = True  # hard rail per cost policy

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# CORS allowlist
ALLOWED_ORIGINS = {
    "https://mikedmote52.github.io",
    "http://localhost:8219",
    "http://127.0.0.1:8219",
    "http://localhost:5050",
    "http://127.0.0.1:5050",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}

try:
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    HAS_YTT = True
except ImportError:
    HAS_YTT = False

# ---- Gemini (direct video understanding via OpenRouter) ----
GEMINI_FLASH = os.environ.get("LEARNENGINE_MODEL_GEMINI_FLASH", "google/gemini-2.5-flash")
GEMINI_PRO = os.environ.get("LEARNENGINE_MODEL_GEMINI_PRO", "google/gemini-2.5-pro")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # optional direct-Google fallback

# In-memory cache keyed by YouTube video ID. Lives for the function lifetime
# (Vercel may recycle). Best-effort speedup for repeat analyses.
_YT_CACHE = {}
_YT_CACHE_MAX = 32


# ============== CORS ==============

def _cors_headers(origin):
    headers = {
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "600",
        "Vary": "Origin",
    }
    allowed = origin in ALLOWED_ORIGINS or bool(
        re.match(r"^http://(localhost|127\.0\.0\.1):\d+$", origin or "")
    )
    if allowed:
        headers["Access-Control-Allow-Origin"] = origin
    return headers


@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin", "")
    for k, v in _cors_headers(origin).items():
        resp.headers[k] = v
    return resp


@app.route("/api/<path:_>", methods=["OPTIONS"])
@app.route("/api", methods=["OPTIONS"])
def options_route(_=None):
    return make_response("", 204)


# ============== LLM helpers ==============

def _openrouter_chat(model, messages, max_tokens=4000, stream=False):
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if stream:
        payload["stream"] = True

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENROUTER_KEY,
            "HTTP-Referer": "https://mikedmote52.github.io/LearnEngine/",
            "X-Title": "LearnEngine",
        },
    )

    if stream:
        return urllib.request.urlopen(req, timeout=120)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else str(e)
        raise RuntimeError("OpenRouter " + str(e.code) + ": " + err_body)


def call_llm(prompt, model, max_tokens=4000, cached_context=None):
    """One-shot LLM call. If cached_context is provided, the transcript/context block
    gets a cache_control breakpoint so OpenRouter reuses it across follow-up calls
    (cuts repeat-context cost ~90%)."""
    if cached_context:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": cached_context,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": prompt},
            ],
        }]
    else:
        messages = [{"role": "user", "content": prompt}]
    return _openrouter_chat(model, messages, max_tokens=max_tokens)


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not extract JSON", text, 0)


def extract_video_id(url):
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract video ID from: " + url)


# ============== Routes ==============

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "has_api_key": bool(OPENROUTER_KEY),
        "models": {
            "haiku": HAIKU,
            "sonnet": SONNET,
            "embed": EMBED_MODEL,
            "never_use_opus": NEVER_USE_OPUS,
        },
    })


@app.route("/api/transcript", methods=["POST"])
def fetch_transcript_route():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not HAS_YTT:
        return jsonify({"error": "Transcript service unavailable"}), 500
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=["en"])
        text_parts = [s.text for s in fetched]
        full_text = " ".join(text_parts)
        return jsonify({
            "video_id": video_id,
            "full_text": full_text,
            "word_count": len(full_text.split()),
        })
    except Exception as e:
        return jsonify({
            "error": "Transcript fetch failed: " + str(e) + ". Try pasting manually.",
        }), 400


# ============== Gemini direct-video helper ==============
# YouTube blocks Vercel's outbound IPs from transcript scraping, but Gemini
# accesses YouTube via Google's own infrastructure when the URL is passed as
# multimodal input — sidesteps the IP block entirely.

def _gemini_youtube_call(prompt, youtube_url, model, max_tokens=12000, timeout=55):
    """Call OpenRouter -> Gemini with a YouTube URL as multimodal input.
    Tries two payload shapes (Google-native file/file_data, then OpenAI-style
    video_url) and returns (text, format_used, model_used) on first success.
    Raises RuntimeError with all attempt errors joined if every shape fails.
    """
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    formats = [
        ("file", [
            {"type": "text", "text": prompt},
            {"type": "file", "file": {"file_data": youtube_url, "mime_type": "video/youtube"}},
        ]),
        ("video_url", [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": youtube_url}},
        ]),
    ]

    errors = []
    for fmt_name, content in formats:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + OPENROUTER_KEY,
                "HTTP-Referer": "https://mikedmote52.github.io/LearnEngine/",
                "X-Title": "LearnEngine",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                msg = data.get("choices", [{}])[0].get("message", {})
                content_out = msg.get("content", "")
                if isinstance(content_out, list):
                    content_out = "".join(
                        p.get("text", "") for p in content_out if isinstance(p, dict)
                    )
                content_out = (content_out or "").strip()
                if not content_out:
                    errors.append(model + "/" + fmt_name + ": empty content")
                    continue
                return content_out, fmt_name, model
        except urllib.error.HTTPError as e:
            err_body = (e.read().decode() if e.fp else str(e))[:600]
            errors.append(model + "/" + fmt_name + ": HTTP " + str(e.code) + ": " + err_body)
            continue
        except urllib.error.URLError as e:
            errors.append(model + "/" + fmt_name + ": URL " + str(e))
            continue
        except Exception as e:
            errors.append(model + "/" + fmt_name + ": " + type(e).__name__ + ": " + str(e))
            continue
    raise RuntimeError("; ".join(errors) or "all formats failed")


def _gemini_direct_call(prompt, youtube_url, model_id, timeout=55):
    """Direct Google Generative Language API fallback (no OpenRouter).
    Only used if both OpenRouter formats fail on both Gemini models AND a
    GEMINI_API_KEY env var is configured."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured for direct fallback")
    # Strip "google/" prefix for direct API
    short_id = model_id.split("/", 1)[-1]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + short_id + ":generateContent?key=" + GEMINI_API_KEY
    )
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"file_data": {"file_uri": youtube_url, "mime_type": "video/youtube"}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {"maxOutputTokens": 12000, "temperature": 0.4},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            cands = data.get("candidates") or []
            if not cands:
                raise RuntimeError("Direct Gemini: no candidates returned")
            parts = cands[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            if not text.strip():
                raise RuntimeError("Direct Gemini: empty text")
            return text.strip(), "google_direct", model_id
    except urllib.error.HTTPError as e:
        err_body = (e.read().decode() if e.fp else str(e))[:600]
        raise RuntimeError("Direct Gemini HTTP " + str(e.code) + ": " + err_body)


# ---- Analysis: two-pass design ----
# Pass 1 (Sonnet, small output ~2k tokens): summary, concepts, fact-check, objectives.
#   Hard reasoning. Concept synthesis, difficulty calibration, fact-checking.
# Pass 2 (Haiku, larger output ~8k tokens): quiz questions + distractors + hints.
#   Bulk formatting. Cached transcript reused from pass 1 (cache hit, ~90% input savings).
# Both calls use cache_control: ephemeral on the transcript so the second pass reads
# the cache instead of paying full input cost again.

@app.route("/api/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    title = data.get("title", "")
    learner_context = data.get("learner_context") or {}

    if not transcript or len(transcript) < 50:
        return jsonify({"error": "Transcript too short"}), 400

    truncated = transcript[:80000]

    learner_section = ""
    if learner_context:
        style = learner_context.get("learning_style") or {}
        teaching_mode = style.get("teaching_mode", "scaffolded")
        weak = learner_context.get("weak_areas") or []
        accuracy = learner_context.get("overall_accuracy", 50)
        learner_section = (
            "\nLEARNER PROFILE:\n"
            "- Accuracy: " + str(accuracy) + "%, Mode: " + teaching_mode + ", "
            "Weak: " + (", ".join(weak) if weak else "None") + "\n"
            "- foundational: simpler language, analogies. "
            "scaffolded: sequential questions. "
            "challenging: synthesis questions."
        )

    cached_transcript = (
        "You are an expert educational content designer.\n"
        "VIDEO TITLE: " + title + "\n"
        "TRANSCRIPT:\n" + truncated + "\n"
    )

    # ---- Pass 1: Sonnet — concept synthesis + fact-check (hard reasoning) ----
    # Lean output: keep explanations under 40 words; nothing longer than necessary.
    # Quiz body comes from Pass 2 (Haiku) so we don't duplicate work here.
    pass1_prompt = (
        learner_section + "\n\n"
        "Analyze the cached transcript above. Respond with COMPACT JSON only "
        "(no markdown fencing, no prose before/after):\n\n"
        "{\n"
        '  "summary": "3-4 sentence summary",\n'
        '  "key_concepts": [\n'
        '    {"id":"slug","name":"Short name","explanation":"Under 30 words",'
        '"topic":"Category","importance":"high|medium|low"}\n'
        "  ],\n"
        '  "fact_check": [\n'
        '    {"claim":"specific claim","assessment":"accurate|partially_accurate|inaccurate|unverifiable",'
        '"correction":"only if inaccurate, else null"}\n'
        "  ],\n"
        '  "misinformation_flags": [],\n'
        '  "difficulty_level": "beginner|intermediate|advanced",\n'
        '  "learning_objectives": ["By the end..."]\n'
        "}\n\n"
        "RULES:\n"
        "- 5-8 key_concepts. Keep each explanation under 30 words.\n"
        "- Fact-check 3-6 specific claims (dates, statistics, named entities).\n"
        "- Keep total output under 1500 tokens. Be terse."
    )

    # ---- Pass 2: Haiku — quiz batch (bulk formatting straight from transcript) ----
    # We run Pass 1 (Sonnet) and Pass 2 (Haiku) in parallel because Haiku doesn't
    # need Pass 1's output — both consume the same cached transcript. Total wall time
    # = max(pass1, pass2) instead of sum, comfortably under Vercel's 60s ceiling.
    pass2_prompt = (
        learner_section + "\n\n"
        "Using the cached transcript above as ground truth, "
        "generate a quiz of 10-12 questions in ONE batched response. JSON only:\n\n"
        "{\n"
        '  "quiz": [\n'
        '    {"question":"Clear question testing understanding",'
        '"concept_id":"slug","concept_name":"readable name","topic":"topic area",'
        '"difficulty":"easy/medium/hard","bloom_level":"remember/understand/apply/analyze/evaluate",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],'
        '"explanation":"Why correct is correct (2-3 sentences)",'
        '"common_misconception":"Most common mistake",'
        '"deeper_insight":"Beyond the video","hint":"Nudge without revealing"}\n'
        "  ]\n"
        "}\n\n"
        "RULES:\n"
        "- 10-12 questions covering the substantive content\n"
        "- Bloom's: 20% remember, 30% understand, 25% apply, 15% analyze, 10% evaluate\n"
        "- Wrong options = REAL misconceptions, not obviously wrong\n"
        "- Exactly ONE option per question has \"correct\": true\n"
        "- Every correct answer MUST be verifiable from the transcript\n"
        "- Every hint guides thinking, doesn't reveal the answer"
    )

    def run_pass1():
        return call_llm(pass1_prompt, model=SONNET, max_tokens=2000, cached_context=cached_transcript)

    def run_pass2():
        return call_llm(pass2_prompt, model=HAIKU, max_tokens=6500, cached_context=cached_transcript)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(run_pass1)
            f2 = ex.submit(run_pass2)
            analysis_text = f1.result(timeout=55)
            quiz_text = f2.result(timeout=55)
        analysis = parse_json_response(analysis_text)
        quiz_payload = parse_json_response(quiz_text)
    except concurrent.futures.TimeoutError:
        return jsonify({"error": "Analysis timed out (>55s)."}), 504
    except json.JSONDecodeError as e:
        return jsonify({"error": "Failed to parse analysis JSON: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Analysis failed: " + str(e)}), 500

    analysis["quiz"] = quiz_payload.get("quiz", [])
    return jsonify(analysis)


# ---- YouTube direct-video analysis (Gemini multimodal) ----
# Replaces the broken transcript-scrape -> /api/analyze flow for YouTube URLs.
# Gemini watches the video via Google's infrastructure and returns concepts,
# fact-checks, and a quiz in a single call. The pasted-transcript flow at
# /api/analyze remains as a manual fallback.

@app.route("/api/analyze-youtube", methods=["POST"])
def analyze_youtube_route():
    data = request.get_json(silent=True) or {}
    youtube_url = (data.get("youtube_url") or data.get("url") or "").strip()
    options = data.get("options") or {}

    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400

    try:
        video_id = extract_video_id(youtube_url)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    canonical_url = "https://www.youtube.com/watch?v=" + video_id

    # In-memory cache hit
    if video_id in _YT_CACHE:
        cached = dict(_YT_CACHE[video_id])
        cached["cached"] = True
        return jsonify(cached)

    learner_context = options.get("learner_context") or {}
    title_hint = (options.get("title") or "").strip()

    learner_section = ""
    if learner_context:
        style = learner_context.get("learning_style") or {}
        teaching_mode = style.get("teaching_mode", "scaffolded")
        weak = learner_context.get("weak_areas") or []
        accuracy = learner_context.get("overall_accuracy", 50)
        learner_section = (
            "\nLEARNER PROFILE:\n"
            "- Accuracy: " + str(accuracy) + "%, Mode: " + teaching_mode + ", "
            "Weak: " + (", ".join(weak) if weak else "None") + "\n"
            "- foundational: simpler language, analogies. "
            "scaffolded: sequential questions. "
            "challenging: synthesis questions.\n"
        )

    prompt = (
        "You are an expert educational content designer. Watch the YouTube video "
        "provided as input, understand its substantive content (concepts, claims, "
        "arguments, examples, evidence), and produce a structured learning analysis."
        + learner_section
        + ("\nVIDEO TITLE HINT: " + title_hint + "\n" if title_hint else "")
        + "\n"
        "Respond with COMPACT JSON only — no markdown fences, no prose before/after.\n\n"
        "{\n"
        '  "summary": "3-4 sentence summary of the video content",\n'
        '  "title": "Best inferred title (or hint if provided)",\n'
        '  "key_concepts": [\n'
        '    {"id":"slug","name":"Short name","explanation":"Under 30 words",'
        '"topic":"Category","importance":"high|medium|low"}\n'
        "  ],\n"
        '  "fact_check": [\n'
        '    {"claim":"specific verifiable claim from the video",'
        '"assessment":"accurate|partially_accurate|inaccurate|unverifiable",'
        '"correction":"only if inaccurate, else null"}\n'
        "  ],\n"
        '  "misinformation_flags": [],\n'
        '  "difficulty_level": "beginner|intermediate|advanced",\n'
        '  "difficulty_score": 5,\n'
        '  "learning_objectives": ["By the end the learner will..."],\n'
        '  "quiz": [\n'
        '    {"question":"Clear question testing real understanding",'
        '"concept_id":"slug","concept_name":"name","topic":"topic",'
        '"difficulty":"easy|medium|hard","bloom_level":"remember|understand|apply|analyze|evaluate",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],'
        '"explanation":"Why correct (2-3 sentences)",'
        '"common_misconception":"Most common mistake",'
        '"deeper_insight":"Beyond the video","hint":"Nudge without revealing"}\n'
        "  ]\n"
        "}\n\n"
        "RULES:\n"
        "- 5-8 key_concepts. Each explanation under 30 words.\n"
        "- 3-6 fact_check items: specific dates, statistics, named entities, causal claims.\n"
        "- 8-12 quiz questions covering the substantive content. Mix Bloom levels: "
        "20% remember, 30% understand, 25% apply, 15% analyze, 10% evaluate.\n"
        "- Wrong options must be REAL misconceptions, never obviously wrong.\n"
        "- Exactly ONE option per question has \"correct\": true.\n"
        "- Every correct answer must be verifiable from the video content.\n"
        "- difficulty_score is an integer 1-10 matching difficulty_level.\n"
        "- Output JSON only — no markdown, no prose, no commentary."
    )

    text = None
    fmt_used = None
    model_used = None
    errors = []

    # Try OpenRouter -> gemini-2.5-flash, then -> gemini-2.5-pro.
    # Long videos can need 90s+ so we allow generous per-call timeouts; the
    # Vercel function maxDuration is set to 300s in vercel.json.
    for model in (GEMINI_FLASH, GEMINI_PRO):
        try:
            text, fmt_used, model_used = _gemini_youtube_call(
                prompt, canonical_url, model=model, max_tokens=12000, timeout=240
            )
            break
        except Exception as e:
            errors.append(str(e))
            continue

    # Direct Google Gemini API fallback (only if GEMINI_API_KEY is configured)
    if not text and GEMINI_API_KEY:
        for model in (GEMINI_FLASH, GEMINI_PRO):
            try:
                text, fmt_used, model_used = _gemini_direct_call(
                    prompt, canonical_url, model_id=model, timeout=240
                )
                break
            except Exception as e:
                errors.append(str(e))
                continue

    if not text:
        return jsonify({
            "error": "Gemini video analysis failed across all model+format combinations.",
            "details": errors,
            "video_id": video_id,
        }), 502

    try:
        analysis = parse_json_response(text)
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Gemini returned non-JSON output: " + str(e),
            "raw_excerpt": text[:1500],
            "video_id": video_id,
            "model_used": model_used,
        }), 500

    # Sanitize quiz: keep only well-formed questions, ensure exactly one correct.
    if isinstance(analysis.get("quiz"), list):
        cleaned = []
        for q in analysis["quiz"]:
            if not isinstance(q, dict):
                continue
            opts = q.get("options") or []
            if not q.get("question") or not opts:
                continue
            correct_count = sum(1 for o in opts if isinstance(o, dict) and o.get("correct"))
            if correct_count == 0:
                continue
            if correct_count > 1:
                found = False
                for o in opts:
                    if isinstance(o, dict) and o.get("correct"):
                        if found:
                            o["correct"] = False
                        else:
                            found = True
            for idx, o in enumerate(opts):
                if isinstance(o, dict) and not o.get("label"):
                    o["label"] = chr(65 + idx)
            cleaned.append(q)
        analysis["quiz"] = cleaned

    # Spec aliases (keeps the new endpoint compatible with both the existing
    # client renderer and the spec's preferred field names).
    analysis["concepts"] = analysis.get("key_concepts", [])
    analysis["fact_checks"] = analysis.get("fact_check", [])
    if "difficulty_score" not in analysis or not isinstance(analysis.get("difficulty_score"), (int, float)):
        diff_map = {"beginner": 3, "intermediate": 6, "advanced": 9}
        analysis["difficulty_score"] = diff_map.get(
            (analysis.get("difficulty_level") or "intermediate").lower(), 5
        )

    quiz_questions = []
    for q in analysis.get("quiz", []):
        opts = q.get("options") or []
        correct_idx = next(
            (i for i, o in enumerate(opts) if isinstance(o, dict) and o.get("correct")),
            0,
        )
        quiz_questions.append({
            "stem": q.get("question", ""),
            "options": [
                (o.get("text", "") if isinstance(o, dict) else str(o)) for o in opts
            ],
            "correct_index": correct_idx,
            "explanation": q.get("explanation", ""),
        })
    analysis["quiz_questions"] = quiz_questions

    analysis["video_id"] = video_id
    analysis["model_used"] = model_used
    analysis["format_used"] = fmt_used

    # LRU-ish cache eviction
    if len(_YT_CACHE) >= _YT_CACHE_MAX:
        try:
            _YT_CACHE.pop(next(iter(_YT_CACHE)))
        except StopIteration:
            pass
    _YT_CACHE[video_id] = analysis

    return jsonify(analysis)


# ============== Podcast resolution + analysis (Phase A) ==============
# Public-API path that doesn't require user accounts:
#   1. Apple Podcasts URL  -> iTunes Search lookup -> RSS feed -> match episode -> MP3
#   2. RSS feed URL        -> parse XML directly
#   3. Direct MP3/M4A URL  -> use as-is
#   4. Spotify show URL    -> scrape og:title, then iTunes search by show name -> RSS
# Spotify-exclusive episodes return EXCLUSIVE_NO_RSS so the client can show a
# helpful message rather than an opaque error.

USER_AGENT = "Mozilla/5.0 LearnEngine/2.0"

# In-memory caches (warm-instance only)
_PODCAST_CACHE = {}
_PODCAST_CACHE_MAX = 16


def _http_get(url, headers=None, timeout=20, max_bytes=None):
    """Simple HTTP GET with size cap. Returns bytes."""
    req = urllib.request.Request(url, headers=headers or {"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if max_bytes:
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise RuntimeError("Response exceeds max_bytes=" + str(max_bytes))
            return data
        return resp.read()


def _itunes_lookup_by_id(podcast_id):
    """Apple Podcasts ID -> feedUrl via iTunes Search API (no key needed)."""
    url = "https://itunes.apple.com/lookup?id=" + str(podcast_id) + "&entity=podcast"
    raw = _http_get(url, timeout=10)
    data = json.loads(raw.decode("utf-8", errors="ignore"))
    results = data.get("results") or []
    if not results:
        raise RuntimeError("iTunes lookup returned no results for id " + str(podcast_id))
    return results[0]


def _itunes_search_by_name(name, limit=5):
    url = "https://itunes.apple.com/search?term=" + urllib.parse.quote(name) + "&entity=podcast&limit=" + str(limit)
    raw = _http_get(url, timeout=10)
    data = json.loads(raw.decode("utf-8", errors="ignore"))
    return data.get("results") or []


def _parse_apple_url(url):
    """Apple Podcasts URL -> (podcast_id, episode_id_optional)."""
    # Patterns:
    #   https://podcasts.apple.com/us/podcast/show-name/id12345
    #   https://podcasts.apple.com/us/podcast/show-name/id12345?i=67890
    m = re.search(r"/id(\d+)", url)
    pid = m.group(1) if m else None
    m2 = re.search(r"[?&]i=(\d+)", url)
    eid = m2.group(1) if m2 else None
    return pid, eid


def _fetch_rss(feed_url):
    """Fetch RSS XML, returns ElementTree root."""
    # 60 MB cap — Tim Ferriss is ~30 MB, large catalog feeds can exceed 25 MB.
    raw = _http_get(feed_url, timeout=30, max_bytes=60_000_000)
    # Some feeds have BOMs/declarations issues; let ET handle it
    try:
        return ET.fromstring(raw)
    except ET.ParseError as e:
        # Try to strip leading whitespace / BOM
        text = raw.decode("utf-8", errors="ignore").lstrip("﻿").strip()
        return ET.fromstring(text)


def _rss_episodes(root):
    """Yield episodes from an RSS root: list of dicts with title, audio_url, guid, pubDate."""
    out = []
    # iTunes namespace not strictly needed; <enclosure url="..."> is standard
    for item in root.iter("item"):
        title_el = item.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        guid_el = item.find("guid")
        guid = (guid_el.text or "").strip() if guid_el is not None else ""
        pub_el = item.find("pubDate")
        pub = (pub_el.text or "").strip() if pub_el is not None else ""
        enc = item.find("enclosure")
        audio_url = enc.attrib.get("url") if (enc is not None) else None
        if audio_url:
            out.append({"title": title, "audio_url": audio_url, "guid": guid, "pubDate": pub})
    return out


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


def _scrape_og_meta(url):
    """Pull og:title and og:description from a page. Spotify episode pages include
    the show name in og:description as 'Show Name · Episode'. Returns dict or {}."""
    try:
        raw = _http_get(url, headers={"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
                        timeout=10, max_bytes=400_000)
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return {}
    out = {}
    m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', text, flags=re.I)
    if m:
        out["og_title"] = m.group(1)
    m = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', text, flags=re.I)
    if m:
        out["og_description"] = m.group(1)
    if "og_title" not in out:
        m = re.search(r'<title>([^<]+)</title>', text, flags=re.I)
        if m:
            out["og_title"] = m.group(1).strip()
    return out


def _scrape_og_title(url):
    """Backwards-compatible single-field helper."""
    return (_scrape_og_meta(url) or {}).get("og_title")


def _spotify_oembed(url):
    """Use Spotify's public oEmbed endpoint for reliable episode/show titles.
    Returns dict with at least {'title': ...} or None on failure."""
    try:
        oembed_url = "https://open.spotify.com/oembed?url=" + urllib.parse.quote(url, safe=":/?&=")
        raw = _http_get(oembed_url, headers={"User-Agent": _BROWSER_UA}, timeout=10, max_bytes=200_000)
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return None


def _spotify_resolve(url):
    """Resolve an open.spotify.com episode/show URL to {feed_url, episode, show_title}.
    Tries oEmbed (reliable) + og:description (carries 'Show · Episode'), then
    iTunes Search to locate the public RSS feed. Raises ValueError with
    EXCLUSIVE_NO_RSS code if no public RSS path can be found."""
    oembed = _spotify_oembed(url) or {}
    meta = _scrape_og_meta(url) or {}
    episode_title = (oembed.get("title") or meta.get("og_title") or "").strip()
    # og:description on Spotify episodes is reliably "Show Name · Episode"
    show_name = ""
    desc = (meta.get("og_description") or "").strip()
    if desc and ("·" in desc or " - " in desc.lower()):
        # Split on the middle-dot first; iOS sometimes substitutes other separators
        parts = re.split(r"\s*[·•|]\s*", desc, maxsplit=1)
        if parts and parts[0]:
            show_name = parts[0].strip()
    # Also check for "Show on Spotify" pattern in og:title for show URLs
    if not show_name and episode_title:
        m = re.match(r"^(.*?)\s*[-|]\s*(?:Listen on Spotify|Podcast on Spotify|Spotify)\s*$", episode_title, flags=re.I)
        if m:
            show_name = m.group(1).strip()
    if not (episode_title or show_name):
        raise ValueError({"code": "EXCLUSIVE_NO_RSS", "msg": "Spotify page is not publicly readable"})
    # Search iTunes by show name (preferred) then by episode title (fallback)
    queries = []
    if show_name:
        queries.append(show_name)
    if episode_title and episode_title not in queries:
        queries.append(episode_title)
    seen_feeds = set()
    last_err = None
    for q in queries:
        try:
            candidates = _itunes_search_by_name(q, limit=5)
        except Exception as e:
            last_err = e
            continue
        for cand in candidates:
            feed = cand.get("feedUrl")
            if not feed or feed in seen_feeds:
                continue
            seen_feeds.add(feed)
            try:
                root = _fetch_rss(feed)
                eps = _rss_episodes(root)
            except Exception as e:
                last_err = e
                continue
            if not eps:
                continue
            # Try strict title match, then substring match using strongest title we have
            target = (episode_title or "").lower()
            match = None
            if target:
                match = next((e for e in eps if (e["title"] or "").strip().lower() == target), None)
                if not match:
                    match = next((e for e in eps if target in (e["title"] or "").lower()), None)
            episode = match or eps[0]
            return {
                "feed_url": feed,
                "episode": episode,
                "show_title": cand.get("collectionName") or show_name or "",
            }
    raise ValueError({
        "code": "EXCLUSIVE_NO_RSS",
        "msg": "appears to be a Spotify exclusive — no public RSS feed found",
    })


def resolve_podcast(url):
    """Resolve any podcast URL -> {feed_url, episode: {audio_url, title, guid}, show_title}.
    Raises ValueError with code 'EXCLUSIVE_NO_RSS' for Spotify-exclusives or unresolvable links."""
    url = (url or "").strip()
    if not url:
        raise ValueError("Empty URL")

    lower = url.lower()
    # Strip query string for path-extension checks (.mp3?awCollectionId=...)
    path_only = lower.split("?", 1)[0]

    # Direct audio file (path ends in known audio extension)
    if path_only.endswith(".mp3") or path_only.endswith(".m4a") or path_only.endswith(".wav") or path_only.endswith(".ogg"):
        return {"feed_url": None, "episode": {"audio_url": url, "title": url.rsplit("/", 1)[-1].split("?")[0], "guid": url}, "show_title": ""}

    # RSS feed (heuristic: ends in xml/rss or contains /feed)
    if path_only.endswith(".xml") or path_only.endswith(".rss") or "/rss" in lower or "/feed" in lower or "feeds." in lower:
        try:
            root = _fetch_rss(url)
            eps = _rss_episodes(root)
            if not eps:
                raise RuntimeError("RSS feed has no episodes with audio enclosure")
            chan = root.find("channel/title")
            show = (chan.text or "") if chan is not None else ""
            return {"feed_url": url, "episode": eps[0], "show_title": show.strip()}
        except Exception as e:
            raise ValueError("Could not parse RSS feed: " + str(e))

    # Apple Podcasts
    if "podcasts.apple.com" in lower:
        pid, eid = _parse_apple_url(url)
        if not pid:
            raise ValueError("Apple Podcasts URL missing podcast id")
        info = _itunes_lookup_by_id(pid)
        feed = info.get("feedUrl")
        show = info.get("collectionName") or info.get("trackName") or ""
        if not feed:
            raise ValueError("Apple Podcasts entry has no public feedUrl")
        root = _fetch_rss(feed)
        eps = _rss_episodes(root)
        if not eps:
            raise ValueError("Resolved RSS has no episodes")
        # If we have an episode track id, try to match by title via iTunes lookup by track id
        if eid:
            try:
                track_info = _itunes_lookup_by_id(eid)
                track_name = (track_info.get("trackName") or "").strip().lower()
                if track_name:
                    match = next((e for e in eps if e["title"].strip().lower() == track_name), None)
                    if match:
                        return {"feed_url": feed, "episode": match, "show_title": show}
            except Exception:
                pass
        return {"feed_url": feed, "episode": eps[0], "show_title": show}

    # Spotify (only the canonical web app domain — not CDN subdomains like byspotify.com)
    if "open.spotify.com" in lower:
        return _spotify_resolve(url)

    # Default: try treating it as an RSS feed
    try:
        root = _fetch_rss(url)
        eps = _rss_episodes(root)
        if eps:
            chan = root.find("channel/title")
            show = (chan.text or "") if chan is not None else ""
            return {"feed_url": url, "episode": eps[0], "show_title": show.strip()}
    except Exception:
        pass

    raise ValueError({"code": "EXCLUSIVE_NO_RSS", "msg": "could not resolve podcast — paste an RSS feed or Apple Podcasts URL"})


AUDIO_CAP_BYTES = 18_000_000  # 18 MB MP3 → ~24 MB base64; stays under Gemini inline limit
AUDIO_HARD_CAP  = 60_000_000  # don't download more than this even when truncating


def _download_audio_capped(audio_url, cap_bytes=AUDIO_CAP_BYTES, hard_cap=AUDIO_HARD_CAP):
    """Stream-download an audio URL, truncating to cap_bytes if larger.
    Returns (bytes, was_truncated, total_size_seen)."""
    req = urllib.request.Request(audio_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        chunks = []
        seen = 0
        truncated = False
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            seen += len(chunk)
            if seen <= cap_bytes:
                chunks.append(chunk)
            else:
                # We've collected enough, just count remaining to report total size
                truncated = True
                # Stop reading after we've seen up to hard_cap so we don't wait forever
                if seen > hard_cap:
                    break
        return b"".join(chunks), truncated, seen


def _gemini_audio_call(prompt, audio_url, model, timeout=240, max_tokens=12000):
    """Call OpenRouter -> Gemini with audio bytes (base64 inline).
    Audio is downloaded server-side and capped at AUDIO_CAP_BYTES so we stay
    under Gemini's inline-data request limit (~20 MB). For long episodes the
    first ~30-45 minutes of content is analyzed.
    Returns (text, format_used).
    """
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    # Download audio (capped)
    audio_bytes, truncated, total = _download_audio_capped(audio_url)
    if not audio_bytes:
        raise RuntimeError("audio download produced 0 bytes")

    # Detect MIME from URL extension; default to mpeg
    lower = audio_url.lower().split("?")[0]
    if lower.endswith(".m4a"):
        mime = "audio/mp4"
    elif lower.endswith(".wav"):
        mime = "audio/wav"
    elif lower.endswith(".ogg"):
        mime = "audio/ogg"
    else:
        mime = "audio/mpeg"

    b64 = base64.b64encode(audio_bytes).decode("ascii")
    data_uri = "data:" + mime + ";base64," + b64

    note = ""
    if truncated:
        note = ("\n\n[Note: episode is large; analyzing the first ~"
                + str(round(len(audio_bytes) / 1_000_000)) + "MB of audio "
                "out of ~" + str(round(total / 1_000_000)) + "MB total. "
                "Focus on the substantive content covered in this portion.]")

    # Two content shapes — try Google-native file_data first, fall back to OpenAI input_audio
    formats = [
        ("file_data_inline", [
            {"type": "text", "text": prompt + note},
            {"type": "file", "file": {"file_data": data_uri, "mime_type": mime}},
        ]),
        ("input_audio", [
            {"type": "text", "text": prompt + note},
            {"type": "input_audio", "input_audio": {"data": b64, "format": mime.split("/")[-1]}},
        ]),
    ]

    errors = []
    for fmt_name, content in formats:
        payload = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + OPENROUTER_KEY,
                "HTTP-Referer": "https://mikedmote52.github.io/LearnEngine/",
                "X-Title": "LearnEngine",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                msg = data.get("choices", [{}])[0].get("message", {})
                text = msg.get("content", "")
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                text = (text or "").strip()
                if text:
                    return text, fmt_name
                errors.append(fmt_name + ": empty content")
        except urllib.error.HTTPError as e:
            err_body = (e.read().decode() if e.fp else str(e))[:500]
            errors.append(fmt_name + ": HTTP " + str(e.code) + ": " + err_body)
        except Exception as e:
            errors.append(fmt_name + ": " + type(e).__name__ + ": " + str(e))

    raise RuntimeError("; ".join(errors) or "all audio formats failed")


def _podcast_quiz_prompt(episode_title, show_title):
    title_hint = (episode_title or "") + (" — " + show_title if show_title else "")
    return (
        "You are an expert educational content designer. Listen to the podcast audio "
        "provided as input. Identify the substantive content (concepts, claims, "
        "arguments, examples, evidence) and produce a structured learning analysis.\n\n"
        + ("EPISODE: " + title_hint + "\n\n" if title_hint else "")
        + "Respond with COMPACT JSON only — no markdown fences, no prose before/after.\n\n"
        "{\n"
        '  "summary": "3-4 sentence summary of the episode",\n'
        '  "title": "Best inferred episode title",\n'
        '  "key_concepts": [\n'
        '    {"id":"slug","name":"Short name","explanation":"Under 30 words",'
        '"topic":"Category","importance":"high|medium|low"}\n'
        "  ],\n"
        '  "fact_check": [\n'
        '    {"claim":"specific verifiable claim from the episode",'
        '"assessment":"accurate|partially_accurate|inaccurate|unverifiable",'
        '"correction":"only if inaccurate, else null"}\n'
        "  ],\n"
        '  "misinformation_flags": [],\n'
        '  "difficulty_level": "beginner|intermediate|advanced",\n'
        '  "learning_objectives": ["By the end the listener will..."],\n'
        '  "quiz": [\n'
        '    {"question":"Clear question testing real understanding",'
        '"concept_id":"slug","concept_name":"name","topic":"topic",'
        '"difficulty":"easy|medium|hard","bloom_level":"remember|understand|apply|analyze|evaluate",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],'
        '"explanation":"Why correct (2-3 sentences)",'
        '"common_misconception":"Most common mistake",'
        '"deeper_insight":"Beyond the episode","hint":"Nudge without revealing"}\n'
        "  ]\n"
        "}\n\n"
        "RULES:\n"
        "- 5-8 key_concepts, each explanation under 30 words.\n"
        "- 3-6 fact_check items: dates, statistics, named entities, causal claims.\n"
        "- 8-12 quiz questions covering substantive content. Mix Bloom levels.\n"
        "- Wrong options must be REAL plausible misconceptions.\n"
        "- Exactly ONE option per question has \"correct\": true.\n"
        "- Output JSON only — no markdown, no prose, no commentary."
    )


@app.route("/api/analyze-podcast", methods=["POST"])
def analyze_podcast_route():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    cache_key = url.split("?")[0][:200]
    if cache_key in _PODCAST_CACHE:
        cached = dict(_PODCAST_CACHE[cache_key])
        cached["cached"] = True
        return jsonify(cached)

    try:
        info = resolve_podcast(url)
    except ValueError as ve:
        payload = ve.args[0] if ve.args and isinstance(ve.args[0], dict) else None
        if payload and payload.get("code") == "EXCLUSIVE_NO_RSS":
            is_spotify = "open.spotify.com" in (url or "").lower()
            if is_spotify:
                return jsonify({
                    "error": "spotify_exclusive",
                    "message": "This episode is Spotify-exclusive and can't be analyzed. "
                               "Try a podcast with an RSS feed (Apple Podcasts, RSS link, "
                               "or direct MP3).",
                    "reason": payload.get("msg", "no public RSS feed found"),
                }), 422
            return jsonify({
                "error": "podcast_not_resolvable",
                "message": "We couldn't find a public RSS feed for this podcast. Try the "
                           "Apple Podcasts URL, the show's RSS feed link, or a direct "
                           "MP3 link.",
                "reason": payload.get("msg", "no public RSS feed found"),
            }), 422
        return jsonify({
            "error": "podcast_not_resolvable",
            "message": "We couldn't resolve that podcast URL. Try the Apple Podcasts URL "
                       "or the show's RSS feed link.",
            "reason": str(ve),
        }), 422

    episode = info["episode"]
    show = info.get("show_title", "")
    audio_url = episode["audio_url"]
    prompt = _podcast_quiz_prompt(episode.get("title", ""), show)

    text = None
    fmt_used = None
    model_used = None
    errors = []
    for model in (GEMINI_FLASH, GEMINI_PRO):
        try:
            text, fmt_used = _gemini_audio_call(prompt, audio_url, model=model, timeout=240)
            model_used = model
            break
        except Exception as e:
            errors.append(model + ": " + str(e))
            continue

    if not text:
        return jsonify({
            "error": "Audio analysis failed across all model attempts.",
            "details": errors[:5],
            "audio_url": audio_url,
        }), 502

    try:
        analysis = parse_json_response(text)
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Gemini returned non-JSON output: " + str(e),
            "raw_excerpt": text[:1500],
            "model_used": model_used,
        }), 500

    # Sanitize quiz the same way YouTube does
    if isinstance(analysis.get("quiz"), list):
        cleaned = []
        for q in analysis["quiz"]:
            if not isinstance(q, dict):
                continue
            opts = q.get("options") or []
            if not q.get("question") or not opts:
                continue
            correct_count = sum(1 for o in opts if isinstance(o, dict) and o.get("correct"))
            if correct_count == 0:
                continue
            if correct_count > 1:
                found = False
                for o in opts:
                    if isinstance(o, dict) and o.get("correct"):
                        if found:
                            o["correct"] = False
                        else:
                            found = True
            for idx, o in enumerate(opts):
                if isinstance(o, dict) and not o.get("label"):
                    o["label"] = chr(65 + idx)
            cleaned.append(q)
        analysis["quiz"] = cleaned

    analysis["concepts"] = analysis.get("key_concepts", [])
    analysis["fact_checks"] = analysis.get("fact_check", [])
    analysis["episode_id"] = re.sub(r"[^a-zA-Z0-9]+", "_", (episode.get("guid") or audio_url))[:48]
    analysis["episode_title"] = analysis.get("title") or episode.get("title", "")
    analysis["show_title"] = show
    analysis["audio_url"] = audio_url
    analysis["feed_url"] = info.get("feed_url")
    analysis["model_used"] = model_used
    analysis["format_used"] = fmt_used

    if len(_PODCAST_CACHE) >= _PODCAST_CACHE_MAX:
        try:
            _PODCAST_CACHE.pop(next(iter(_PODCAST_CACHE)))
        except StopIteration:
            pass
    _PODCAST_CACHE[cache_key] = analysis

    return jsonify(analysis)


# ============== Book chapter analysis (Phase B) ==============

def _book_prompt(chapter_title):
    title_line = ("\nCHAPTER TITLE: " + chapter_title + "\n") if chapter_title else "\n"
    return (
        "You are an expert educational content designer. The user has provided "
        "photos of consecutive pages from a book chapter. Read the pages, identify "
        "the substantive content (concepts, claims, arguments, equations, diagrams), "
        "and produce a structured learning analysis."
        + title_line + "\n"
        "Respond with COMPACT JSON only — no markdown fences, no prose before/after.\n\n"
        "{\n"
        '  "summary": "3-4 sentence summary of the chapter content",\n'
        '  "title": "Best inferred chapter title",\n'
        '  "key_concepts": [\n'
        '    {"id":"slug","name":"Short name","explanation":"Under 30 words",'
        '"topic":"Category","importance":"high|medium|low"}\n'
        "  ],\n"
        '  "fact_check": [\n'
        '    {"claim":"specific verifiable claim from the chapter",'
        '"assessment":"accurate|partially_accurate|inaccurate|unverifiable",'
        '"correction":"only if inaccurate, else null"}\n'
        "  ],\n"
        '  "difficulty_level": "beginner|intermediate|advanced",\n'
        '  "learning_objectives": ["By the end the reader will..."],\n'
        '  "quiz": [\n'
        '    {"question":"Question that references the actual content of the pages",'
        '"concept_id":"slug","concept_name":"name","topic":"topic",'
        '"difficulty":"easy|medium|hard","bloom_level":"remember|understand|apply|analyze|evaluate",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],'
        '"explanation":"Why correct (2-3 sentences)",'
        '"common_misconception":"Most common mistake",'
        '"deeper_insight":"Beyond the chapter","hint":"Nudge without revealing"}\n'
        "  ]\n"
        "}\n\n"
        "RULES:\n"
        "- 5-8 key_concepts, each explanation under 30 words.\n"
        "- 3-6 fact_check items: dates, statistics, named entities, causal claims.\n"
        "- 8-12 quiz questions that REFERENCE the actual chapter content.\n"
        "- Wrong options must be plausible misconceptions, not obviously wrong.\n"
        "- Exactly ONE option per question has \"correct\": true.\n"
        "- Output JSON only — no markdown fences, no prose, no commentary."
    )


def _gemini_book_call(prompt, image_data_urls, model, timeout=240, max_tokens=12000):
    """Call OpenRouter -> Gemini with multiple page images. Returns (text, format_used)."""
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    # Build content array: prompt first, then images.
    # Use OpenAI-style image_url format which OpenRouter passes through.
    content = [{"type": "text", "text": prompt}]
    for du in image_data_urls:
        if isinstance(du, str) and du.startswith("data:"):
            content.append({"type": "image_url", "image_url": {"url": du}})
        elif isinstance(du, str) and (du.startswith("http://") or du.startswith("https://")):
            content.append({"type": "image_url", "image_url": {"url": du}})
        else:
            # Assume bare base64 string -> wrap as data URI (jpeg)
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + du}})

    payload = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENROUTER_KEY,
            "HTTP-Referer": "https://mikedmote52.github.io/LearnEngine/",
            "X-Title": "LearnEngine",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            msg = data.get("choices", [{}])[0].get("message", {})
            text = msg.get("content", "")
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            text = (text or "").strip()
            if not text:
                raise RuntimeError("empty content")
            return text, "image_url"
    except urllib.error.HTTPError as e:
        err_body = (e.read().decode() if e.fp else str(e))[:600]
        raise RuntimeError("HTTP " + str(e.code) + ": " + err_body)


@app.route("/api/analyze-book-chapter", methods=["POST"])
def analyze_book_chapter_route():
    data = request.get_json(silent=True) or {}
    images = data.get("images") or []
    chapter_title = (data.get("chapter_title") or "").strip()

    if not isinstance(images, list) or not images:
        return jsonify({"error": "images array is required"}), 400
    if len(images) > 30:
        return jsonify({"error": "max 30 images per request"}), 400

    prompt = _book_prompt(chapter_title)

    text = None
    fmt_used = None
    model_used = None
    errors = []
    for model in (GEMINI_FLASH, GEMINI_PRO):
        try:
            text, fmt_used = _gemini_book_call(prompt, images, model=model, timeout=240)
            model_used = model
            break
        except Exception as e:
            errors.append(model + ": " + str(e))
            continue

    if not text:
        return jsonify({
            "error": "Book chapter analysis failed across all model attempts.",
            "details": errors[:5],
        }), 502

    try:
        analysis = parse_json_response(text)
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Gemini returned non-JSON output: " + str(e),
            "raw_excerpt": text[:1500],
            "model_used": model_used,
        }), 500

    # Sanitize quiz
    if isinstance(analysis.get("quiz"), list):
        cleaned = []
        for q in analysis["quiz"]:
            if not isinstance(q, dict):
                continue
            opts = q.get("options") or []
            if not q.get("question") or not opts:
                continue
            correct_count = sum(1 for o in opts if isinstance(o, dict) and o.get("correct"))
            if correct_count == 0:
                continue
            if correct_count > 1:
                found = False
                for o in opts:
                    if isinstance(o, dict) and o.get("correct"):
                        if found:
                            o["correct"] = False
                        else:
                            found = True
            for idx, o in enumerate(opts):
                if isinstance(o, dict) and not o.get("label"):
                    o["label"] = chr(65 + idx)
            cleaned.append(q)
        analysis["quiz"] = cleaned

    analysis["concepts"] = analysis.get("key_concepts", [])
    analysis["fact_checks"] = analysis.get("fact_check", [])
    analysis["chapter_title"] = chapter_title or analysis.get("title", "")
    analysis["page_count"] = len(images)
    analysis["model_used"] = model_used
    analysis["format_used"] = fmt_used

    return jsonify(analysis)


# ============== Due-count sync (for Mac-side cron) ==============
# Stores the most recently reported due count in /tmp on the warm function
# instance. POST writes; GET reads. Used by the daily reminder script on
# Mike's Mac to decide whether to fire an iMessage at 8 AM.

_DUE_FILE = "/tmp/learnengine_due_count.json"


@app.route("/api/sync-due-count", methods=["POST", "GET"])
def sync_due_count_route():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        try:
            count = int(data.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        ts = data.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        client = (data.get("client") or "")[:32]
        try:
            with open(_DUE_FILE, "w") as f:
                json.dump({"count": count, "ts": ts, "client": client, "stored_at": time.time()}, f)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "count": count, "ts": ts})

    # GET
    try:
        with open(_DUE_FILE, "r") as f:
            payload = json.load(f)
        return jsonify(payload)
    except FileNotFoundError:
        return jsonify({"count": 0, "ts": None, "client": None, "stored_at": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Adaptive follow-up (Sonnet) ----

@app.route("/api/followup", methods=["POST"])
def followup_route():
    data = request.get_json(silent=True) or {}
    wrong_answers = data.get("wrong_answers") or []
    weak_concepts = data.get("weak_concepts") or []
    transcript = data.get("transcript", "")
    learner_context = data.get("learner_context") or {}

    style_ctx = ""
    if learner_context:
        mode = (learner_context.get("learning_style") or {}).get("teaching_mode", "scaffolded")
        style_ctx = "\nLEARNER MODE: " + mode

    cached = "TRANSCRIPT (use as source of truth):\n" + transcript[:40000]

    # Two-pass split, run IN PARALLEL like /api/analyze:
    # Pass 1 (Sonnet, ~1200 tok): diagnosis + teaching strategy + focus areas
    # Pass 2 (Haiku, ~8000 tok): 6-10 batched teaching questions, working
    #   directly off the wrong answers (doesn't depend on Sonnet's diag output)
    # Total wall time ~= max(pass1, pass2) instead of sum -> roughly halves the
    # user-facing latency (was 50-90s sequential, now 25-45s).
    diag_prompt = (
        "You are an adaptive tutor diagnosing a struggling learner. JSON ONLY, no prose:\n\n"
        "STRUGGLED WITH: " + json.dumps(weak_concepts) + "\n"
        "WRONG ANSWERS: " + json.dumps(wrong_answers) + style_ctx + "\n\n"
        "Return:\n"
        '{"diagnosis":"1-2 sentence misconception pattern",'
        '"teaching_strategy":"1-2 sentences on how follow-up questions will fix it",'
        '"focus_areas":["concept_id or short name", "..."]}\n'
        "Be terse. Total under 250 tokens. JSON only — no markdown, no prose."
    )

    quiz_prompt = (
        "You are an adaptive tutor. A learner just answered these questions wrong. "
        "Build a 6-10 question TEACHING follow-up that re-teaches the underlying concepts "
        "from a more basic angle, scaffolding upward. JSON ONLY, no prose.\n\n"
        "WRONG ANSWERS (each shows the question, what they picked, the correct answer, "
        "why they were wrong):\n"
        + json.dumps(wrong_answers) + style_ctx + "\n\n"
        "Return:\n"
        '{"quiz":[{"question":"...","concept_id":"...","concept_name":"...","topic":"...",'
        '"difficulty":"easy|medium|hard","bloom_level":"remember|understand|apply|analyze",'
        '"scaffold_note":"why this question now","teaching_moment":"the insight it lands",'
        '"builds_on_concept":"which missed concept this builds on",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],"explanation":"why correct is correct","deeper_insight":"beyond the basics",'
        '"hint":"nudge without revealing"}]}\n\n'
        "RULES:\n"
        "- 6-10 questions. Sequence: Q1-2 prerequisites (easy), Q3-5 build (medium), "
        "Q6-8 apply (hard), Q9-10 synthesize.\n"
        "- Each question must directly target one of the missed concepts above.\n"
        "- Exactly ONE option per question has \"correct\": true.\n"
        "- Wrong options must be REAL plausible misconceptions a learner could hold.\n"
        "- JSON only. No markdown fences. No prose."
    )

    def run_diag():
        return call_llm(diag_prompt, model=SONNET, max_tokens=600, cached_context=cached if transcript else None)

    def run_quiz():
        return call_llm(quiz_prompt, model=SONNET, max_tokens=8000, cached_context=cached if transcript else None)

    diag = {"diagnosis": "", "teaching_strategy": "", "focus_areas": weak_concepts or []}
    quiz_payload = {"quiz": []}
    diag_err = None
    quiz_err = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_diag = ex.submit(run_diag)
            f_quiz = ex.submit(run_quiz)
            try:
                diag_text = f_diag.result(timeout=180)
                diag = parse_json_response(diag_text)
            except Exception as e:
                diag_err = str(e)
            try:
                quiz_text = f_quiz.result(timeout=240)
                quiz_payload = parse_json_response(quiz_text)
            except Exception as e:
                quiz_err = str(e)
    except Exception as e:
        return jsonify({"error": "Followup orchestration failed: " + str(e)}), 500

    quiz = quiz_payload.get("quiz", []) or []

    # Sanitize quiz: drop malformed questions, ensure exactly one correct option
    cleaned = []
    for q in quiz:
        if not isinstance(q, dict):
            continue
        opts = q.get("options") or []
        if not q.get("question") or not opts:
            continue
        correct_count = sum(1 for o in opts if isinstance(o, dict) and o.get("correct"))
        if correct_count == 0:
            # No correct flagged — promote first option as a graceful fallback so we
            # still return useful content rather than dropping the question.
            if isinstance(opts[0], dict):
                opts[0]["correct"] = True
                correct_count = 1
            else:
                continue
        if correct_count > 1:
            seen = False
            for o in opts:
                if isinstance(o, dict) and o.get("correct"):
                    if seen:
                        o["correct"] = False
                    else:
                        seen = True
        for idx, o in enumerate(opts):
            if isinstance(o, dict) and not o.get("label"):
                o["label"] = chr(65 + idx)
        cleaned.append(q)

    if not cleaned:
        # The Sonnet quiz pass either failed or returned nothing usable.
        # Return a clear error so the client surfaces it instead of hanging.
        msg_parts = []
        if quiz_err:
            msg_parts.append("quiz pass: " + quiz_err)
        if diag_err:
            msg_parts.append("diag pass: " + diag_err)
        return jsonify({
            "error": "Follow-up generation produced no usable questions. "
                     + ("; ".join(msg_parts) if msg_parts else "Try again."),
        }), 502

    return jsonify({
        "diagnosis": diag.get("diagnosis", "") if isinstance(diag, dict) else "",
        "teaching_strategy": diag.get("teaching_strategy", "") if isinstance(diag, dict) else "",
        "focus_areas": (diag.get("focus_areas", []) if isinstance(diag, dict) else []) or weak_concepts,
        "quiz": cleaned,
    })


# ---- Spaced review (Haiku — bulk formatting from due-list) ----

@app.route("/api/review", methods=["POST"])
def review_route():
    data = request.get_json(silent=True) or {}
    due = data.get("due_concepts") or []
    if not due:
        return jsonify({"message": "No concepts due for review"}), 200

    prompt = (
        "Generate a spaced repetition review quiz. Test from DIFFERENT ANGLES than before. "
        "Low mastery = easier. High mastery = harder.\n\n"
        "CONCEPTS DUE: " + json.dumps(due[:15]) + "\n\n"
        "JSON only:\n"
        "{\n"
        '  "focus_areas":["topics"],\n'
        '  "quiz":[{"question":"...","concept_id":"...","concept_name":"...","topic":"...",'
        '"difficulty":"easy/medium/hard","bloom_level":"...",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],"explanation":"...","deeper_insight":"...","hint":"..."}]\n'
        "}"
    )

    try:
        text = call_llm(prompt, model=HAIKU, max_tokens=8000)
        return jsonify(parse_json_response(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Embeddings (similarity-based spaced repetition) ----

@app.route("/api/embed", methods=["POST"])
def embed_route():
    data = request.get_json(silent=True) or {}
    inputs = data.get("inputs")
    if not inputs:
        return jsonify({"error": "inputs is required"}), 400
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list):
        return jsonify({"error": "inputs must be string or list of strings"}), 400
    if len(inputs) > 200:
        return jsonify({"error": "max 200 inputs per request"}), 400

    body = json.dumps({"model": EMBED_MODEL, "input": inputs}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENROUTER_KEY,
            "HTTP-Referer": "https://mikedmote52.github.io/LearnEngine/",
            "X-Title": "LearnEngine",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return jsonify({
                "model": data.get("model", EMBED_MODEL),
                "embeddings": [d["embedding"] for d in data["data"]],
                "usage": data.get("usage"),
            })
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else str(e)
        return jsonify({"error": "Embeddings " + str(e.code) + ": " + err_body}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Streaming explanation (Haiku, capped) ----

@app.route("/api/explain", methods=["POST"])
def explain_route():
    data = request.get_json(silent=True) or {}
    concept = (data.get("concept") or "").strip()
    if not concept:
        return jsonify({"error": "concept required"}), 400
    transcript_excerpt = (data.get("transcript") or "")[:8000]

    prompt = (
        "Explain this concept clearly and concisely (under 180 words):\n\n"
        "CONCEPT: " + concept + "\n\n"
        "TRANSCRIPT EXCERPT (for grounding):\n" + transcript_excerpt + "\n\n"
        "Use a short analogy. End with one question that checks understanding."
    )

    def gen():
        try:
            resp = _openrouter_chat(
                HAIKU,
                [{"role": "user", "content": prompt}],
                max_tokens=400,
                stream=True,
            )
            for line in resp:
                if not line:
                    continue
                if line.startswith(b"data: "):
                    chunk = line[6:].strip()
                    if chunk == b"[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as e:
            yield "\n[error: " + str(e) + "]"

    return Response(gen(), mimetype="text/plain")
