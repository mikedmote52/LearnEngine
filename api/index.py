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
import concurrent.futures
from flask import Flask, request, jsonify, make_response, Response
import urllib.request
import urllib.error

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

    # Try OpenRouter -> gemini-2.5-flash, then -> gemini-2.5-pro
    for model in (GEMINI_FLASH, GEMINI_PRO):
        try:
            text, fmt_used, model_used = _gemini_youtube_call(
                prompt, canonical_url, model=model, max_tokens=12000, timeout=55
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
                    prompt, canonical_url, model_id=model, timeout=55
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

    prompt = (
        "You are an adaptive tutor. A learner struggled. TEACH through questions, don't just retest.\n\n"
        "STRUGGLED WITH: " + json.dumps(weak_concepts) + "\n"
        "WRONG ANSWERS: " + json.dumps(wrong_answers) + style_ctx + "\n\n"
        "Generate 6-10 TEACHING questions in ONE batched response. JSON only:\n"
        "{\n"
        '  "diagnosis":"Misconception pattern detected",\n'
        '  "teaching_strategy":"How this fixes the misunderstanding",\n'
        '  "focus_areas":["concepts retested"],\n'
        '  "quiz":[{"question":"...","concept_id":"...","concept_name":"...","topic":"...",'
        '"difficulty":"easy/medium/hard","bloom_level":"remember/understand/apply/analyze",'
        '"scaffold_note":"...","teaching_moment":"...",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],"explanation":"...","deeper_insight":"...","hint":"..."}]\n'
        "}\n"
        "Sequence: Q1-2 prerequisites (easy), Q3-5 build (medium), Q6-8 apply (hard), Q9-10 synthesize"
    )

    # Followup uses the same two-pass split:
    # Pass 1 (Sonnet, ~1500 tok): diagnosis + teaching strategy
    # Pass 2 (Haiku, ~7000 tok): the 6-10 batched teaching questions
    try:
        diag_text = call_llm(
            prompt + "\n\nFor THIS pass, return ONLY: "
            '{"diagnosis":"...","teaching_strategy":"...","focus_areas":["..."]}',
            model=SONNET, max_tokens=1500, cached_context=cached,
        )
        diag = parse_json_response(diag_text)
    except Exception as e:
        return jsonify({"error": "Followup pass 1 failed: " + str(e)}), 500

    quiz_prompt = (
        "Build the teaching-question batch for the diagnosis below. JSON only:\n\n"
        + "DIAGNOSIS: " + json.dumps(diag) + "\n\n"
        + "WRONG ANSWERS: " + json.dumps(wrong_answers) + style_ctx + "\n\n"
        '{"quiz":[{"question":"...","concept_id":"...","concept_name":"...","topic":"...",'
        '"difficulty":"easy/medium/hard","bloom_level":"remember/understand/apply/analyze",'
        '"scaffold_note":"...","teaching_moment":"...",'
        '"options":['
        '{"label":"A","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"B","text":"...","correct":true,"why_wrong":null},'
        '{"label":"C","text":"...","correct":false,"why_wrong":"..."},'
        '{"label":"D","text":"...","correct":false,"why_wrong":"..."}'
        '],"explanation":"...","deeper_insight":"...","hint":"..."}]}\n\n'
        "Sequence: Q1-2 prerequisites (easy), Q3-5 build (medium), Q6-8 apply (hard), Q9-10 synthesize. "
        "6-10 questions total. Exactly ONE correct option per question."
    )
    try:
        quiz_text = call_llm(
            quiz_prompt, model=HAIKU, max_tokens=8000, cached_context=cached
        )
        quiz_payload = parse_json_response(quiz_text)
    except Exception as e:
        return jsonify({"error": "Followup pass 2 failed: " + str(e)}), 500

    return jsonify({
        "diagnosis": diag.get("diagnosis", ""),
        "teaching_strategy": diag.get("teaching_strategy", ""),
        "focus_areas": diag.get("focus_areas", []),
        "quiz": quiz_payload.get("quiz", []),
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
