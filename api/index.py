"""
LearnEngine - Vercel Serverless API
Stateless API: transcript fetching + Claude analysis via OpenRouter.
All learner state lives client-side in IndexedDB.
"""

import json
import os
import re
from flask import Flask, request, jsonify
import urllib.request
import urllib.error

app = Flask(__name__)

MODEL = os.environ.get("LEARNENGINE_MODEL", "anthropic/claude-sonnet-4")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_YTT = True
except ImportError:
    HAS_YTT = False


def call_llm(prompt, max_tokens=12000):
    """Call LLM via OpenRouter (preferred) or Anthropic API."""
    if OPENROUTER_KEY:
        return _call_openrouter(prompt, max_tokens)
    elif ANTHROPIC_KEY:
        return _call_anthropic(prompt, max_tokens)
    else:
        raise Exception("No API key configured. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in Vercel environment variables.")


def _call_openrouter(prompt, max_tokens):
    """Call Claude via OpenRouter (OpenAI-compatible API)."""
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "HTTP-Referer": "https://learnengine.vercel.app",
            "X-Title": "LearnEngine",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise Exception(f"OpenRouter API error ({e.code}): {error_body}")


def _call_anthropic(prompt, max_tokens):
    """Call Anthropic API directly."""
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise Exception(f"Anthropic API error ({e.code}): {error_body}")


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not extract JSON", text, 0)


def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


# ==== Routes ====

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "has_api_key": bool(OPENROUTER_KEY or ANTHROPIC_KEY),
        "provider": "openrouter" if OPENROUTER_KEY else ("anthropic" if ANTHROPIC_KEY else "none"),
    })


@app.route("/api/transcript", methods=["POST"])
def fetch_transcript_route():
    data = request.json or {}
    url = data.get("url", "").strip()
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
        segments = []
        text_parts = []
        for snippet in fetched:
            segments.append({"start": snippet.start, "duration": snippet.duration, "text": snippet.text})
            text_parts.append(snippet.text)
        full_text = " ".join(text_parts)
        return jsonify({"video_id": video_id, "full_text": full_text, "word_count": len(full_text.split())})
    except Exception as e:
        return jsonify({"error": f"Transcript fetch failed: {str(e)}. Try pasting the transcript manually."}), 400


@app.route("/api/analyze", methods=["POST"])
def analyze_route():
    data = request.json or {}
    transcript = data.get("transcript", "").strip()
    title = data.get("title", "")
    learner_context = data.get("learner_context", {})

    if not transcript or len(transcript) < 50:
        return jsonify({"error": "Transcript too short"}), 400

    truncated = transcript[:80000]

    learner_section = ""
    if learner_context:
        style = learner_context.get("learning_style", {})
        teaching_mode = style.get("teaching_mode", "scaffolded")
        weak = learner_context.get("weak_areas", [])
        accuracy = learner_context.get("overall_accuracy", 50)
        learner_section = f"""
LEARNER PROFILE:
- Accuracy: {accuracy}%, Mode: {teaching_mode}, Weak: {', '.join(weak) if weak else 'None'}
- If foundational: simpler language, analogies. If scaffolded: sequential questions. If challenging: synthesis questions."""

    prompt = f"""You are an expert educational content designer. Analyze this transcript for DEEP UNDERSTANDING.

VIDEO TITLE: {title}
{learner_section}
TRANSCRIPT:
{truncated}

Respond with JSON only (no markdown fencing):

{{
  "summary": "3-5 sentence summary",
  "key_concepts": [
    {{"id": "concept_1", "name": "Short name", "explanation": "Clear explanation", "simple_analogy": "Everyday analogy", "topic": "Category", "importance": "high/medium/low"}}
  ],
  "fact_check": [
    {{"claim": "Specific claim", "assessment": "accurate/partially_accurate/inaccurate/unverifiable", "correction": "If inaccurate, null if accurate.", "reasoning": "Why"}}
  ],
  "misinformation_flags": [
    {{"statement": "...", "issue": "...", "severity": "high/medium/low"}}
  ],
  "bias_notes": "Notable biases or missing context",
  "difficulty_level": "beginner/intermediate/advanced",
  "learning_objectives": ["By the end you should be able to..."],
  "quiz": [
    {{
      "question": "Clear question testing understanding",
      "concept_id": "which concept", "concept_name": "readable name", "topic": "topic area",
      "difficulty": "easy/medium/hard", "bloom_level": "remember/understand/apply/analyze/evaluate",
      "options": [
        {{"label": "A", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "B", "text": "...", "correct": true, "why_wrong": null}},
        {{"label": "C", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "D", "text": "...", "correct": false, "why_wrong": "..."}}
      ],
      "explanation": "Why correct answer is correct",
      "common_misconception": "Most common mistake",
      "deeper_insight": "Beyond the video",
      "hint": "Nudge without revealing"
    }}
  ]
}}

RULES:
- 10-15 questions covering ALL key concepts
- Bloom's: 20% remember, 30% understand, 25% apply, 15% analyze, 10% evaluate
- Wrong options = REAL misconceptions, not obviously wrong
- Fact-check ALL claims, dates, statistics
- Every hint guides thinking, doesn't reveal answer"""

    try:
        text = call_llm(prompt, max_tokens=12000)
        return jsonify(parse_json_response(text))
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse AI response: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/followup", methods=["POST"])
def followup_route():
    data = request.json or {}
    wrong_answers = data.get("wrong_answers", [])
    weak_concepts = data.get("weak_concepts", [])
    transcript = data.get("transcript", "")
    learner_context = data.get("learner_context", {})

    style_ctx = ""
    if learner_context:
        mode = learner_context.get("learning_style", {}).get("teaching_mode", "scaffolded")
        style_ctx = f"\nLEARNER MODE: {mode}"

    prompt = f"""You are an adaptive tutor. A learner struggled. TEACH through questions, don't just retest.

STRUGGLED WITH: {json.dumps(weak_concepts, indent=2)}
WRONG ANSWERS: {json.dumps(wrong_answers, indent=2)}
{style_ctx}
TRANSCRIPT: {transcript[:40000]}

Generate 6-10 TEACHING questions. JSON only:
{{
  "diagnosis": "Misconception pattern detected",
  "teaching_strategy": "How this fixes the misunderstanding",
  "focus_areas": ["concepts retested"],
  "quiz": [
    {{
      "question": "...", "concept_id": "...", "concept_name": "...", "topic": "...",
      "difficulty": "easy/medium/hard", "bloom_level": "remember/understand/apply/analyze",
      "scaffold_note": "How this builds understanding",
      "teaching_moment": "Key insight",
      "options": [
        {{"label": "A", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "B", "text": "...", "correct": true, "why_wrong": null}},
        {{"label": "C", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "D", "text": "...", "correct": false, "why_wrong": "..."}}
      ],
      "explanation": "Rich explanation with analogy",
      "deeper_insight": "Additional context",
      "hint": "Thinking prompt"
    }}
  ]
}}
Sequence: Q1-2 prerequisites (easy), Q3-5 build (medium), Q6-8 apply (hard), Q9-10 synthesize"""

    try:
        return jsonify(parse_json_response(call_llm(prompt, 10000)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/review", methods=["POST"])
def review_route():
    data = request.json or {}
    due = data.get("due_concepts", [])
    if not due:
        return jsonify({"message": "No concepts due for review"}), 200

    prompt = f"""Generate a spaced repetition review quiz. Test from DIFFERENT ANGLES than before.
Low mastery = easier. High mastery = harder.

CONCEPTS DUE: {json.dumps(due[:15], indent=2)}

JSON only:
{{
  "focus_areas": ["topics"],
  "quiz": [
    {{
      "question": "...", "concept_id": "...", "concept_name": "...", "topic": "...",
      "difficulty": "easy/medium/hard", "bloom_level": "...",
      "options": [
        {{"label": "A", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "B", "text": "...", "correct": true, "why_wrong": null}},
        {{"label": "C", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "D", "text": "...", "correct": false, "why_wrong": "..."}}
      ],
      "explanation": "...", "deeper_insight": "...", "hint": "..."
    }}
  ]
}}"""

    try:
        return jsonify(parse_json_response(call_llm(prompt, 8000)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
