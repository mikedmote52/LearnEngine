"""
LearnEngine - Adaptive Learning System
Flask web application that processes YouTube videos into interactive quizzes
with fact-checking, spaced repetition, and knowledge gap analysis.

Run: python app.py
Then open: http://localhost:5050
"""

import json
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from transcript import fetch_transcript, extract_video_id
from analyzer import (
    analyze_transcript, generate_followup_quiz, generate_review_quiz,
    save_api_key, get_api_key
)
from db import (
    init_db, save_video, get_all_videos, get_video,
    save_quiz, get_quiz, get_all_quizzes,
    record_quiz_result, get_learning_stats, get_learner_profile,
    get_due_concepts, get_concept_history
)

app = Flask(__name__)

# Initialize database on startup
init_db()

# ---- API Routes ----

@app.route("/api/process", methods=["POST"])
def process_video():
    """Process a YouTube video: fetch transcript, analyze, generate quiz."""
    data = request.json or {}
    url = data.get("url", "").strip()
    title = data.get("title", "").strip()
    manual_transcript = data.get("manual_transcript", "").strip()

    if not url and not manual_transcript:
        return jsonify({"error": "Provide a YouTube URL or paste a transcript"}), 400

    # Validate URL format if provided
    if url:
        try:
            extract_video_id(url)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # Limit transcript size at the API layer
    if manual_transcript and len(manual_transcript) > 500000:
        return jsonify({"error": "Transcript too large. Maximum 500,000 characters."}), 400

    # Step 1: Get transcript
    if manual_transcript:
        video_id = extract_video_id(url) if url else "manual"
        transcript_data = {
            "video_id": video_id,
            "url": url or "manual_paste",
            "full_text": manual_transcript,
            "segments": [],
            "word_count": len(manual_transcript.split()),
            "fetched_at": datetime.now().isoformat(),
        }
    else:
        transcript_data = fetch_transcript(url)
        if transcript_data.get("error"):
            return jsonify({
                "error": f"Transcript fetch failed: {transcript_data['error']}",
                "suggestion": "Try pasting the transcript manually"
            }), 400

    # Get learner context for adaptive quiz generation
    profile = get_learner_profile()
    learner_context = {
        "learning_style": profile.get("learning_style", {}),
        "weak_areas": profile.get("weak_areas", []),
        "strong_areas": profile.get("strong_areas", []),
        "overall_accuracy": round(profile.get("overall_accuracy", 0.5) * 100, 1),
    }

    # Step 2: Analyze with Claude (adapted to learner)
    analysis = analyze_transcript(
        transcript_data["full_text"],
        video_title=title,
        learner_context=learner_context,
    )

    if analysis.get("error"):
        return jsonify({"error": analysis["error"], "transcript": transcript_data}), 500

    # Step 3: Save video data
    video_data = {
        "video_id": transcript_data["video_id"],
        "url": transcript_data["url"],
        "title": title or f"Video {transcript_data['video_id']}",
        "transcript": transcript_data,
        "analysis": analysis,
        "processed_at": datetime.now().isoformat(),
    }
    save_video(video_data)

    # Step 4: Save quiz
    quiz_data = None
    if analysis.get("quiz"):
        quiz_data = {
            "video_id": transcript_data["video_id"],
            "video_title": title or f"Video {transcript_data['video_id']}",
            "quiz_type": "initial",
            "questions": analysis["quiz"],
            "difficulty_level": analysis.get("difficulty_level", "mixed"),
        }
        quiz_id = save_quiz(quiz_data)
        quiz_data["id"] = quiz_id

    return jsonify({
        "video": video_data,
        "quiz": quiz_data,
        "fact_check": analysis.get("fact_check", []),
        "misinformation_flags": analysis.get("misinformation_flags", []),
        "summary": analysis.get("summary", ""),
        "learning_objectives": analysis.get("learning_objectives", []),
        "concept_map": analysis.get("concept_map", []),
    })


@app.route("/api/submit_quiz", methods=["POST"])
def submit_quiz():
    """Submit quiz answers and get results with adaptive feedback."""
    data = request.json or {}
    quiz_id = data.get("quiz_id")
    answers = data.get("answers", {})

    if not quiz_id:
        return jsonify({"error": "quiz_id is required"}), 400

    quiz = get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found"}), 404

    # Grade the quiz
    details = []
    correct_count = 0
    wrong_answers = []

    for i, question in enumerate(quiz["questions"]):
        selected = answers.get(str(i))
        correct_option = next((o for o in question["options"] if o.get("correct")), None)
        if not correct_option:
            continue

        is_correct = selected == correct_option["label"]

        if is_correct:
            correct_count += 1
        else:
            wrong_answers.append({
                "question": question["question"],
                "selected": selected,
                "selected_text": next((o["text"] for o in question["options"] if o["label"] == selected), ""),
                "correct": correct_option["label"],
                "correct_text": correct_option["text"],
                "why_wrong": next((o.get("why_wrong","") for o in question["options"] if o["label"] == selected), ""),
                "concept_id": question.get("concept_id", ""),
                "concept_name": question.get("concept_name", ""),
                "topic": question.get("topic", "general"),
            })

        details.append({
            "question": question["question"],
            "correct": is_correct,
            "selected": selected,
            "correct_answer": correct_option["label"],
            "correct_text": correct_option["text"],
            "explanation": question.get("explanation", ""),
            "deeper_insight": question.get("deeper_insight", ""),
            "common_misconception": question.get("common_misconception", ""),
            "hint": question.get("hint", ""),
            "concept_id": question.get("concept_id", ""),
            "concept_name": question.get("concept_name", ""),
            "topic": question.get("topic", "general"),
            "bloom_level": question.get("bloom_level", ""),
        })

    total = len(quiz["questions"])
    score = correct_count / total if total > 0 else 0
    result = record_quiz_result(quiz_id, answers, score, details)

    return jsonify({
        "score": round(score * 100, 1),
        "correct": correct_count,
        "total": total,
        "details": details,
        "wrong_answers": wrong_answers,
        "weak_areas": result["weak_areas"],
        "strong_areas": result.get("strong_areas", []),
        "due_for_review": result["due_for_review"],
        "learning_style": result.get("learning_style", {}),
        "needs_followup": score < 0.8,
    })


@app.route("/api/followup_quiz", methods=["POST"])
def followup_quiz():
    """Generate a follow-up quiz targeting weak areas with adaptive teaching."""
    data = request.json or {}
    quiz_id = data.get("quiz_id")
    wrong_answers = data.get("wrong_answers", [])

    quiz = get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Original quiz not found"}), 404

    # Get transcript and learner context
    video = get_video(quiz.get("video_id", ""))
    transcript_text = video["transcript"]["full_text"] if video else ""
    profile = get_learner_profile()
    learner_context = {
        "learning_style": profile.get("learning_style", {}),
        "weak_areas": profile.get("weak_areas", []),
        "overall_accuracy": round(profile.get("overall_accuracy", 0.5) * 100, 1),
    }

    weak_concepts = [w.get("concept_id", w.get("topic")) for w in wrong_answers]

    result = generate_followup_quiz(
        weak_concepts, wrong_answers, transcript_text,
        learner_context=learner_context
    )

    if result.get("error"):
        return jsonify({"error": result["error"]}), 500

    followup = {
        "video_id": quiz.get("video_id"),
        "video_title": quiz.get("video_title"),
        "quiz_type": "followup",
        "parent_quiz_id": quiz_id,
        "focus_areas": result.get("focus_areas", weak_concepts),
        "questions": result.get("quiz", []),
        "diagnosis": result.get("diagnosis", ""),
        "teaching_strategy": result.get("teaching_strategy", ""),
    }
    fid = save_quiz(followup)
    followup["id"] = fid

    return jsonify(followup)


@app.route("/api/review_quiz", methods=["POST"])
def review_quiz():
    """Generate a spaced repetition review quiz for due concepts."""
    profile = get_learner_profile()
    due = get_due_concepts(profile)

    if not due:
        return jsonify({"message": "No concepts due for review. You're all caught up!"}), 200

    result = generate_review_quiz(due)

    if result.get("error"):
        return jsonify({"error": result["error"]}), 500

    review = {
        "video_id": "review",
        "video_title": "Spaced Repetition Review",
        "quiz_type": "review",
        "focus_areas": result.get("focus_areas", []),
        "questions": result.get("quiz", []),
    }
    rid = save_quiz(review)
    review["id"] = rid

    return jsonify(review)


@app.route("/api/stats")
def stats():
    """Get learning statistics."""
    return jsonify(get_learning_stats())


@app.route("/api/videos")
def videos():
    """Get all processed videos."""
    return jsonify(get_all_videos())


@app.route("/api/quizzes")
def quizzes():
    """Get all quizzes."""
    return jsonify(get_all_quizzes())


@app.route("/api/due_reviews")
def due_reviews():
    """Get concepts due for spaced repetition review."""
    profile = get_learner_profile()
    due = get_due_concepts(profile)
    return jsonify([
        {"concept_id": cid, **concept}
        for cid, concept in due[:20]
    ])


@app.route("/api/config", methods=["GET", "POST"])
def config():
    """Get or update configuration."""
    if request.method == "POST":
        data = request.json or {}
        if data.get("api_key"):
            try:
                save_api_key(data["api_key"])
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        return jsonify({"status": "saved"})
    else:
        key = get_api_key()
        return jsonify({"has_api_key": bool(key)})


@app.route("/")
def index():
    return render_template_string(FRONTEND_HTML)


# ---- Frontend ----

FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LearnEngine</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a25;
    --border: #2a2a3a;
    --text: #e0e0e8;
    --text-muted: #8888a0;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.15);
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --orange: #f97316;
    --cyan: #06b6d4;
    --radius: 12px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
  }
  .app { max-width: 1100px; margin: 0 auto; padding: 24px; }

  .header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 20px 0; margin-bottom: 32px; border-bottom: 1px solid var(--border);
  }
  .header h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
  .header h1 span { color: var(--accent); }

  .nav { display: flex; gap: 8px; flex-wrap: wrap; }
  .nav button {
    padding: 8px 16px; border: 1px solid var(--border); border-radius: 8px;
    background: var(--surface); color: var(--text-muted); cursor: pointer;
    font-size: 14px; transition: all 0.2s; position: relative;
  }
  .nav button:hover, .nav button.active {
    background: var(--accent-glow); color: var(--accent); border-color: var(--accent);
  }
  .nav .badge {
    position: absolute; top: -6px; right: -6px; background: var(--orange);
    color: white; font-size: 11px; font-weight: 700; padding: 1px 6px;
    border-radius: 10px; min-width: 18px; text-align: center;
  }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; margin-bottom: 16px;
  }
  .card h2 { font-size: 18px; margin-bottom: 16px; font-weight: 600; }
  .card h3 { font-size: 15px; margin-bottom: 12px; font-weight: 600; color: var(--text-muted); }

  .input-group { margin-bottom: 16px; }
  .input-group label { display: block; font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }
  input[type="text"], input[type="password"], textarea {
    width: 100%; padding: 12px 16px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); font-size: 15px; font-family: inherit;
    transition: border-color 0.2s;
  }
  input:focus, textarea:focus { outline: none; border-color: var(--accent); }
  textarea { min-height: 120px; resize: vertical; }

  .btn {
    padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer;
    font-size: 15px; font-weight: 600; transition: all 0.2s;
  }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { background: #5558e6; }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn-secondary:hover { border-color: var(--accent); }
  .btn-sm { padding: 6px 12px; font-size: 13px; }
  .btn-orange { background: var(--orange); color: white; }
  .btn-orange:hover { background: #ea6c0a; }
  .btn-green { background: var(--green); color: white; }
  .btn-green:hover { background: #1aab52; }

  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat-card {
    background: var(--surface2); border-radius: 8px; padding: 16px;
    border: 1px solid var(--border);
  }
  .stat-card .value { font-size: 28px; font-weight: 700; color: var(--accent); }
  .stat-card .label { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

  .question-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; margin-bottom: 16px;
  }
  .question-num { font-size: 12px; color: var(--accent); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .question-text { font-size: 16px; font-weight: 500; margin-bottom: 16px; line-height: 1.5; }
  .option {
    display: block; width: 100%; text-align: left; padding: 12px 16px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); cursor: pointer; margin-bottom: 8px;
    font-size: 14px; transition: all 0.15s;
  }
  .option:hover { border-color: var(--accent); background: var(--accent-glow); }
  .option.selected { border-color: var(--accent); background: var(--accent-glow); }
  .option.correct { border-color: var(--green); background: rgba(34,197,94,0.1); }
  .option.incorrect { border-color: var(--red); background: rgba(239,68,68,0.1); }
  .option .label-badge {
    display: inline-block; width: 24px; height: 24px; line-height: 24px;
    text-align: center; border-radius: 50%; background: var(--surface2);
    font-weight: 600; font-size: 12px; margin-right: 10px;
  }

  .fact-item { padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid; }
  .fact-accurate { border-color: var(--green); background: rgba(34,197,94,0.05); }
  .fact-partially_accurate { border-color: var(--yellow); background: rgba(234,179,8,0.05); }
  .fact-inaccurate { border-color: var(--red); background: rgba(239,68,68,0.05); }
  .fact-unverifiable { border-color: var(--text-muted); background: rgba(136,136,160,0.05); }
  .fact-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }

  .explanation {
    margin-top: 12px; padding: 16px; background: var(--bg);
    border-radius: 8px; border: 1px solid var(--border);
    font-size: 14px; line-height: 1.6;
  }
  .explanation .deeper {
    margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border);
    color: var(--cyan); font-style: italic;
  }
  .explanation .misconception {
    margin-top: 8px; padding: 8px 12px; background: rgba(239,68,68,0.05);
    border-radius: 6px; border-left: 3px solid var(--orange);
    font-size: 13px; color: var(--text-muted);
  }

  .flag-item {
    padding: 12px 16px; margin-bottom: 8px; border-radius: 8px;
    border: 1px solid var(--red); background: rgba(239,68,68,0.05);
  }
  .flag-item.medium { border-color: var(--orange); background: rgba(249,115,22,0.05); }
  .flag-item.low { border-color: var(--yellow); background: rgba(234,179,8,0.05); }

  .progress-bar {
    width: 100%; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden;
  }
  .progress-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }

  .score-display { text-align: center; padding: 32px; }
  .score-big { font-size: 64px; font-weight: 800; }
  .score-big.great { color: var(--green); }
  .score-big.okay { color: var(--yellow); }
  .score-big.needs-work { color: var(--orange); }
  .score-big.struggling { color: var(--red); }

  .loading { text-align: center; padding: 40px; }
  .spinner {
    display: inline-block; width: 32px; height: 32px; border: 3px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .video-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px; background: var(--surface2); border-radius: 8px;
    margin-bottom: 8px; border: 1px solid var(--border);
  }
  .video-item:hover { border-color: var(--accent); }
  .video-title { font-weight: 500; }
  .video-date { font-size: 13px; color: var(--text-muted); }

  .topic-pill {
    display: inline-block; padding: 4px 10px; border-radius: 12px;
    font-size: 12px; margin: 2px; border: 1px solid var(--border);
  }
  .topic-weak { border-color: var(--red); color: var(--red); }
  .topic-strong { border-color: var(--green); color: var(--green); }
  .topic-learning { border-color: var(--yellow); color: var(--yellow); }

  .mastery-bar {
    display: flex; height: 20px; border-radius: 4px; overflow: hidden;
    margin: 8px 0; background: var(--surface2);
  }
  .mastery-bar .segment { height: 100%; transition: width 0.3s; }
  .mastery-mastered { background: var(--green); }
  .mastery-learning { background: var(--yellow); }
  .mastery-struggling { background: var(--red); }

  .hint-box {
    margin-top: 8px; padding: 12px; background: rgba(99,102,241,0.08);
    border-radius: 8px; border: 1px dashed var(--accent);
    font-size: 14px; color: var(--accent); cursor: pointer;
  }

  .teaching-banner {
    padding: 16px; background: rgba(6,182,212,0.08); border: 1px solid var(--cyan);
    border-radius: 8px; margin-bottom: 16px; font-size: 14px;
  }
  .teaching-banner strong { color: var(--cyan); }

  .tab-row { display: flex; gap: 0; margin-bottom: 24px; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 12px 20px; cursor: pointer; font-size: 14px; color: var(--text-muted);
    border-bottom: 2px solid transparent; transition: all 0.2s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .hidden { display: none; }

  @media (max-width: 600px) {
    .app { padding: 12px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<div class="app" id="app"></div>

<script>
const API = '';

// ---- Escape HTML to prevent XSS ----
function esc(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ---- State ----
let state = {
  view: 'dashboard',
  stats: null,
  videos: [],
  quizzes: [],
  currentQuiz: null,
  currentAnswers: {},
  quizResults: null,
  processing: false,
  hasApiKey: false,
  hintsRevealed: {},
  reviewDue: 0,
};

// ---- API Helpers ----
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'API error');
  return data;
}

// ---- Init ----
async function init() {
  try {
    const [stats, videos, quizzes, config] = await Promise.all([
      api('/api/stats'),
      api('/api/videos'),
      api('/api/quizzes'),
      api('/api/config'),
    ]);
    state.stats = stats;
    state.videos = videos;
    state.quizzes = quizzes;
    state.hasApiKey = config.has_api_key;
    state.reviewDue = stats.concepts_due_for_review || 0;
  } catch(e) {
    console.error('Init error:', e);
  }
  render();
}

// ---- Render ----
function render() {
  const app = document.getElementById('app');
  app.innerHTML = `
    ${renderHeader()}
    ${state.view === 'dashboard' ? renderDashboard() : ''}
    ${state.view === 'process' ? renderProcess() : ''}
    ${state.view === 'quiz' ? renderQuiz() : ''}
    ${state.view === 'results' ? renderResults() : ''}
    ${state.view === 'history' ? renderHistory() : ''}
    ${state.view === 'settings' ? renderSettings() : ''}
  `;
}

function renderHeader() {
  return `
    <div class="header">
      <h1><span>Learn</span>Engine</h1>
      <div class="nav">
        <button class="${state.view === 'dashboard' ? 'active' : ''}" onclick="navigate('dashboard')">Dashboard</button>
        <button class="${state.view === 'process' ? 'active' : ''}" onclick="navigate('process')">+ New Video</button>
        <button class="${state.view === 'history' ? 'active' : ''}" onclick="navigate('history')">History</button>
        <button class="${state.view === 'settings' ? 'active' : ''}" onclick="navigate('settings')">Settings</button>
        ${state.reviewDue > 0 ? `
          <button class="btn-orange btn-sm" onclick="startReview()" style="position:relative;">
            Review (${state.reviewDue})
          </button>
        ` : ''}
      </div>
    </div>
  `;
}

function renderDashboard() {
  const s = state.stats || {};
  const totalConcepts = (s.concepts_mastered||0) + (s.concepts_learning||0) + (s.concepts_struggling||0);
  const masteredPct = totalConcepts > 0 ? ((s.concepts_mastered||0)/totalConcepts*100) : 0;
  const learningPct = totalConcepts > 0 ? ((s.concepts_learning||0)/totalConcepts*100) : 0;
  const strugglingPct = totalConcepts > 0 ? ((s.concepts_struggling||0)/totalConcepts*100) : 0;

  return `
    <div class="stats-grid">
      <div class="stat-card">
        <div class="value">${s.total_videos || 0}</div>
        <div class="label">Videos Processed</div>
      </div>
      <div class="stat-card">
        <div class="value">${s.total_quizzes || 0}</div>
        <div class="label">Quizzes Taken</div>
      </div>
      <div class="stat-card">
        <div class="value">${s.overall_accuracy || 0}%</div>
        <div class="label">Overall Accuracy</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color: ${(s.concepts_due_for_review||0) > 0 ? 'var(--orange)' : 'var(--green)'}">
          ${s.concepts_due_for_review || 0}
        </div>
        <div class="label">Due for Review</div>
      </div>
    </div>

    ${totalConcepts > 0 ? `
      <div class="card">
        <h2>Concept Mastery</h2>
        <div class="mastery-bar">
          <div class="segment mastery-mastered" style="width:${masteredPct}%" title="Mastered: ${s.concepts_mastered}"></div>
          <div class="segment mastery-learning" style="width:${learningPct}%" title="Learning: ${s.concepts_learning}"></div>
          <div class="segment mastery-struggling" style="width:${strugglingPct}%" title="Struggling: ${s.concepts_struggling}"></div>
        </div>
        <div style="display:flex; gap:20px; font-size:13px; color:var(--text-muted); margin-top:8px;">
          <span style="color:var(--green);">Mastered: ${s.concepts_mastered||0}</span>
          <span style="color:var(--yellow);">Learning: ${s.concepts_learning||0}</span>
          <span style="color:var(--red);">Struggling: ${s.concepts_struggling||0}</span>
        </div>
      </div>
    ` : ''}

    ${(s.concepts_due_for_review||0) > 0 ? `
      <div class="card" style="border-color: var(--orange);">
        <h2>Review Time</h2>
        <p style="color: var(--text-muted); margin-bottom: 12px;">
          ${s.concepts_due_for_review} concept${s.concepts_due_for_review > 1 ? 's are' : ' is'} due for spaced repetition review.
          Reviewing now strengthens long-term retention.
        </p>
        <button class="btn btn-orange" onclick="startReview()">Start Review Session</button>
      </div>
    ` : ''}

    ${!state.hasApiKey ? `
      <div class="card" style="border-color: var(--yellow);">
        <h2>Setup Required</h2>
        <p style="color: var(--text-muted); margin-bottom: 12px;">Add your Anthropic API key to enable AI-powered analysis and quiz generation.</p>
        <button class="btn btn-primary btn-sm" onclick="navigate('settings')">Configure API Key</button>
      </div>
    ` : ''}

    ${s.weak_areas && s.weak_areas.length ? `
      <div class="card">
        <h2>Areas to Focus On</h2>
        <div>${s.weak_areas.map(a => `<span class="topic-pill topic-weak">${esc(a)}</span>`).join('')}</div>
      </div>
    ` : ''}

    ${s.strong_areas && s.strong_areas.length ? `
      <div class="card">
        <h2>Your Strengths</h2>
        <div>${s.strong_areas.map(a => `<span class="topic-pill topic-strong">${esc(a)}</span>`).join('')}</div>
      </div>
    ` : ''}

    ${state.videos.length ? `
      <div class="card">
        <h2>Recent Videos</h2>
        ${state.videos.slice(0, 5).map(v => `
          <div class="video-item">
            <div>
              <div class="video-title">${esc(v.title || v.video_id)}</div>
              <div class="video-date">${v.processed_at ? new Date(v.processed_at).toLocaleDateString() : ''}</div>
            </div>
            <button class="btn btn-secondary btn-sm" onclick="viewVideoQuiz('${esc(v.video_id)}')">Take Quiz</button>
          </div>
        `).join('')}
      </div>
    ` : `
      <div class="card" style="text-align: center; padding: 48px;">
        <h2>No videos yet</h2>
        <p style="color: var(--text-muted); margin: 12px 0 20px;">Drop a YouTube link to start learning</p>
        <button class="btn btn-primary" onclick="navigate('process')">Process Your First Video</button>
      </div>
    `}
  `;
}

function renderProcess() {
  return `
    <div class="card">
      <h2>Process a Video</h2>
      <div class="input-group">
        <label>YouTube URL</label>
        <input type="text" id="videoUrl" placeholder="https://www.youtube.com/watch?v=..." />
      </div>
      <div class="input-group">
        <label>Video Title (optional, helps with analysis)</label>
        <input type="text" id="videoTitle" placeholder="Title of the video" />
      </div>
      <details style="margin-bottom: 16px;">
        <summary style="cursor: pointer; color: var(--text-muted); font-size: 14px;">Paste transcript manually (if auto-fetch fails)</summary>
        <div class="input-group" style="margin-top: 12px;">
          <textarea id="manualTranscript" placeholder="Paste transcript here..."></textarea>
        </div>
      </details>
      <button class="btn btn-primary" id="processBtn" onclick="processVideo()" ${state.processing ? 'disabled' : ''}>
        ${state.processing ? 'Analyzing... (this takes 15-30 seconds)' : 'Analyze & Generate Quiz'}
      </button>
      <div id="processError" style="color: var(--red); margin-top: 12px;"></div>
    </div>
  `;
}

function renderQuiz() {
  const quiz = state.currentQuiz;
  if (!quiz) return '<div class="card">No quiz loaded</div>';

  const questions = quiz.questions || [];
  const answeredCount = Object.keys(state.currentAnswers).length;

  return `
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
          <h2>${esc(quiz.video_title || 'Quiz')}</h2>
          <p style="color: var(--text-muted); font-size: 14px;">
            ${quiz.quiz_type === 'followup' ? 'Follow-up: Targeting Your Weak Areas' :
              quiz.quiz_type === 'review' ? 'Spaced Repetition Review' :
              questions.length + ' questions'}
            ${quiz.focus_areas && quiz.focus_areas.length ? ' | Focus: ' + quiz.focus_areas.map(a => esc(a)).join(', ') : ''}
          </p>
        </div>
        <div style="font-size: 14px; color: var(--text-muted);">
          ${answeredCount} / ${questions.length} answered
        </div>
      </div>
      <div class="progress-bar" style="margin-bottom: 24px;">
        <div class="progress-fill" style="width: ${(answeredCount / questions.length) * 100}%"></div>
      </div>
    </div>

    ${quiz.diagnosis ? `
      <div class="teaching-banner">
        <strong>Adaptive Teaching Mode</strong><br>
        ${esc(quiz.diagnosis)}<br>
        <em style="color:var(--text-muted)">${esc(quiz.teaching_strategy || '')}</em>
      </div>
    ` : ''}

    ${questions.map((q, i) => `
      <div class="question-card">
        <div class="question-num">
          Question ${i + 1}
          ${q.difficulty ? ' &middot; ' + esc(q.difficulty) : ''}
          ${q.topic ? ' &middot; ' + esc(q.topic) : ''}
          ${q.bloom_level ? ' &middot; ' + esc(q.bloom_level) : ''}
        </div>
        <div class="question-text">${esc(q.question)}</div>
        ${q.scaffold_note ? `<p style="font-size: 13px; color: var(--cyan); margin-bottom: 12px; font-style: italic;">${esc(q.scaffold_note)}</p>` : ''}
        ${q.teaching_moment ? `<p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">${esc(q.teaching_moment)}</p>` : ''}
        <div>
          ${q.options.map(o => `
            <button class="option ${state.currentAnswers[i] === o.label ? 'selected' : ''}"
              onclick="selectAnswer(${i}, '${esc(o.label)}')">
              <span class="label-badge">${esc(o.label)}</span>${esc(o.text)}
            </button>
          `).join('')}
        </div>
        ${q.hint && !state.hintsRevealed[i] ? `
          <div class="hint-box" onclick="revealHint(${i})">Need a hint? Click here.</div>
        ` : ''}
        ${q.hint && state.hintsRevealed[i] ? `
          <div class="hint-box" style="cursor:default;">${esc(q.hint)}</div>
        ` : ''}
      </div>
    `).join('')}

    <div style="text-align: center; padding: 24px;">
      <button class="btn btn-primary" onclick="submitQuiz()"
        ${answeredCount < questions.length ? 'disabled' : ''}>
        Submit Quiz
      </button>
      <p style="font-size: 13px; color: var(--text-muted); margin-top: 8px;">
        ${answeredCount < questions.length
          ? (questions.length - answeredCount) + ' remaining'
          : 'Ready to submit!'}
      </p>
    </div>
  `;
}

function renderResults() {
  const r = state.quizResults;
  if (!r) return '';

  const scoreClass = r.score >= 90 ? 'great' : r.score >= 70 ? 'okay' : r.score >= 50 ? 'needs-work' : 'struggling';
  const quiz = state.currentQuiz;

  return `
    <div class="card score-display">
      <div class="score-big ${scoreClass}">${r.score}%</div>
      <p style="font-size: 18px; margin-top: 8px;">${r.correct} of ${r.total} correct</p>
      ${r.score >= 90 ? `
        <p style="color: var(--green); margin-top: 16px; font-size: 16px;">
          Excellent! Concepts scheduled for spaced review to lock in retention.
        </p>
      ` : r.score >= 70 ? `
        <p style="color: var(--yellow); margin-top: 16px; font-size: 16px;">
          Good foundation. A follow-up quiz will strengthen the gaps.
        </p>
      ` : `
        <p style="color: var(--orange); margin-top: 16px; font-size: 16px;">
          Let's work on this together. The follow-up quiz will teach these concepts differently.
        </p>
      `}
      ${r.needs_followup ? `
        <button class="btn btn-orange" style="margin-top: 20px;" onclick="requestFollowup()">
          Start Adaptive Follow-Up
        </button>
        <p style="font-size: 13px; color: var(--text-muted); margin-top: 8px;">
          The system will analyze your mistakes and create a teaching sequence tailored to your gaps.
        </p>
      ` : ''}
    </div>

    ${r.weak_areas && r.weak_areas.length ? `
      <div class="card">
        <h2>Areas to Strengthen</h2>
        <div>${r.weak_areas.map(a => `<span class="topic-pill topic-weak">${esc(a)}</span>`).join('')}</div>
      </div>
    ` : ''}

    <div class="card">
      <h2>Detailed Review</h2>
      ${r.details.map((d, i) => `
        <div class="question-card" style="border-left: 3px solid ${d.correct ? 'var(--green)' : 'var(--red)'};">
          <div class="question-num">${d.correct ? 'CORRECT' : 'INCORRECT'} &middot; Question ${i + 1}
            ${d.bloom_level ? ' &middot; ' + esc(d.bloom_level) : ''}
          </div>
          <div class="question-text">${esc(d.question)}</div>
          ${!d.correct ? `
            <p style="margin-bottom: 8px;">
              <span style="color: var(--red);">Your answer: ${esc(d.selected)}</span> &middot;
              <span style="color: var(--green);">Correct: ${esc(d.correct_answer)}) ${esc(d.correct_text)}</span>
            </p>
          ` : ''}
          <div class="explanation">
            ${esc(d.explanation)}
            ${d.common_misconception ? `
              <div class="misconception">
                <strong>Common misconception:</strong> ${esc(d.common_misconception)}
              </div>
            ` : ''}
            ${d.deeper_insight ? `<div class="deeper">${esc(d.deeper_insight)}</div>` : ''}
          </div>
        </div>
      `).join('')}
    </div>

    <div style="text-align: center; padding: 24px;">
      <button class="btn btn-secondary" onclick="navigate('dashboard')">Back to Dashboard</button>
    </div>
  `;
}

function renderHistory() {
  return `
    <div class="card">
      <h2>All Videos</h2>
      ${state.videos.length === 0 ? '<p style="color: var(--text-muted);">No videos processed yet.</p>' : ''}
      ${state.videos.map(v => {
        const fc = v.analysis?.fact_check || [];
        const flags = v.analysis?.misinformation_flags || [];
        return `
          <div class="video-item" style="flex-direction: column; align-items: stretch; gap: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <div>
                <div class="video-title">${esc(v.title || v.video_id)}</div>
                <div class="video-date">
                  ${v.processed_at ? new Date(v.processed_at).toLocaleDateString() : ''} &middot;
                  ${v.transcript?.word_count || '?'} words
                  ${flags.length ? ' &middot; <span style="color: var(--red);">' + flags.length + ' flag' + (flags.length>1?'s':'') + '</span>' : ''}
                </div>
              </div>
              <div style="display: flex; gap: 8px;">
                <button class="btn btn-secondary btn-sm" onclick="viewVideoQuiz('${esc(v.video_id)}')">Take Quiz</button>
                <button class="btn btn-secondary btn-sm" onclick="toggleDetail('${esc(v.video_id)}')">Analysis</button>
              </div>
            </div>
            <div id="detail-${esc(v.video_id)}" class="hidden">
              ${v.analysis?.summary ? `<p style="margin-bottom: 16px;">${esc(v.analysis.summary)}</p>` : ''}
              ${v.analysis?.learning_objectives && v.analysis.learning_objectives.length ? `
                <h3>Learning Objectives</h3>
                <div style="margin-bottom:16px;">
                  ${v.analysis.learning_objectives.map(o => `<p style="font-size:14px; color:var(--text-muted); margin-bottom:4px;">&bull; ${esc(o)}</p>`).join('')}
                </div>
              ` : ''}
              ${fc.length ? `
                <h3>Fact Check Results</h3>
                ${fc.map(f => `
                  <div class="fact-item fact-${esc(f.assessment)}">
                    <span class="fact-badge" style="background: ${
                      f.assessment === 'accurate' ? 'rgba(34,197,94,0.15); color: var(--green)' :
                      f.assessment === 'inaccurate' ? 'rgba(239,68,68,0.15); color: var(--red)' :
                      f.assessment === 'partially_accurate' ? 'rgba(234,179,8,0.15); color: var(--yellow)' :
                      'rgba(136,136,160,0.15); color: var(--text-muted)'
                    }">${esc((f.assessment||'').replace('_', ' '))}</span>
                    <p style="margin-top: 8px; font-size: 14px;"><strong>"${esc(f.claim)}"</strong></p>
                    ${f.correction ? `<p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">${esc(f.correction)}</p>` : ''}
                  </div>
                `).join('')}
              ` : ''}
              ${flags.length ? `
                <h3 style="margin-top: 16px;">Misinformation Flags</h3>
                ${flags.map(f => `
                  <div class="flag-item ${esc(f.severity)}">
                    <strong style="font-size: 14px;">${esc(f.statement)}</strong>
                    <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">${esc(f.issue)}</p>
                  </div>
                `).join('')}
              ` : ''}
            </div>
          </div>
        `;
      }).join('')}
    </div>

    <div class="card">
      <h2>All Quizzes</h2>
      ${state.quizzes.length === 0 ? '<p style="color: var(--text-muted);">No quizzes yet.</p>' : ''}
      ${state.quizzes.map(q => `
        <div class="video-item">
          <div>
            <div class="video-title">
              ${esc(q.video_title || q.video_id)}
              ${q.quiz_type === 'followup' ? '<span style="color:var(--orange);font-size:12px;"> (Follow-up)</span>' :
                q.quiz_type === 'review' ? '<span style="color:var(--cyan);font-size:12px;"> (Review)</span>' : ''}
            </div>
            <div class="video-date">${q.created_at ? new Date(q.created_at).toLocaleDateString() : ''} &middot; ${(q.questions||[]).length} questions</div>
          </div>
          <button class="btn btn-secondary btn-sm" onclick="startQuiz('${esc(q.id)}')">Take Quiz</button>
        </div>
      `).join('')}
    </div>
  `;
}

function renderSettings() {
  return `
    <div class="card">
      <h2>API Configuration</h2>
      <p style="color: var(--text-muted); margin-bottom: 16px;">
        Your API key is stored locally and never leaves your machine except to call the Anthropic API directly.
      </p>
      <div class="input-group">
        <label>Anthropic API Key</label>
        <input type="password" id="apiKeyInput" placeholder="sk-ant-..." />
      </div>
      <button class="btn btn-primary" onclick="saveApiKey()">Save API Key</button>
      <div id="settingsMsg" style="margin-top: 12px; font-size: 14px;"></div>
    </div>

    <div class="card">
      <h2>How It Works</h2>
      <div style="color: var(--text-muted); font-size: 14px; line-height: 1.8;">
        <p><strong>1. Process a Video</strong> &mdash; Paste a YouTube URL. The system fetches the transcript and runs it through AI analysis.</p>
        <p style="margin-top: 8px;"><strong>2. AI Analysis</strong> &mdash; Claude extracts key concepts, fact-checks claims, flags misinformation, and generates an adaptive quiz tailored to your current level.</p>
        <p style="margin-top: 8px;"><strong>3. Take the Quiz</strong> &mdash; Questions are sequenced using Bloom's taxonomy, from recall through synthesis. Hints are available if you get stuck.</p>
        <p style="margin-top: 8px;"><strong>4. Review & Learn</strong> &mdash; Every question includes a detailed explanation, common misconceptions, and a deeper insight beyond the video.</p>
        <p style="margin-top: 8px;"><strong>5. Adaptive Follow-ups</strong> &mdash; Score below 80%? The system analyzes your specific misconceptions and generates a teaching sequence that builds understanding step by step.</p>
        <p style="margin-top: 8px;"><strong>6. Spaced Repetition</strong> &mdash; The SM-2 algorithm schedules concept reviews at optimal intervals. When concepts are due, start a review session to strengthen retention.</p>
        <p style="margin-top: 8px;"><strong>7. Learning Profile</strong> &mdash; The system tracks your mastery across topics and adapts difficulty, teaching approach, and explanation style to how you learn best.</p>
      </div>
    </div>
  `;
}

// ---- Actions ----
function navigate(view) {
  state.view = view;
  state.hintsRevealed = {};
  if (view === 'dashboard') init();
  else render();
}

function selectAnswer(qi, label) {
  state.currentAnswers[qi] = label;
  render();
}

function revealHint(qi) {
  state.hintsRevealed[qi] = true;
  render();
}

async function processVideo() {
  const url = document.getElementById('videoUrl')?.value?.trim();
  const title = document.getElementById('videoTitle')?.value?.trim();
  const manual = document.getElementById('manualTranscript')?.value?.trim();

  if (!url && !manual) {
    document.getElementById('processError').textContent = 'Enter a YouTube URL or paste a transcript';
    return;
  }

  state.processing = true;
  render();

  try {
    const result = await api('/api/process', {
      method: 'POST',
      body: { url, title, manual_transcript: manual }
    });

    if (result.quiz) {
      state.currentQuiz = result.quiz;
      state.currentAnswers = {};
      state.hintsRevealed = {};
      state.view = 'quiz';
    }
    state.processing = false;
    render();
  } catch(e) {
    state.processing = false;
    render();
    const errEl = document.getElementById('processError');
    if (errEl) errEl.textContent = e.message;
  }
}

async function submitQuiz() {
  const quiz = state.currentQuiz;
  if (!quiz) return;

  try {
    const result = await api('/api/submit_quiz', {
      method: 'POST',
      body: { quiz_id: quiz.id, answers: state.currentAnswers }
    });
    state.quizResults = result;
    state.view = 'results';
    render();
    window.scrollTo(0, 0);
  } catch(e) {
    alert('Error submitting: ' + e.message);
  }
}

async function requestFollowup() {
  const quiz = state.currentQuiz;
  const results = state.quizResults;
  if (!quiz || !results) return;

  try {
    state.processing = true;
    render();
    const followup = await api('/api/followup_quiz', {
      method: 'POST',
      body: { quiz_id: quiz.id, wrong_answers: results.wrong_answers }
    });
    state.currentQuiz = followup;
    state.currentAnswers = {};
    state.hintsRevealed = {};
    state.processing = false;
    state.view = 'quiz';
    render();
    window.scrollTo(0, 0);
  } catch(e) {
    state.processing = false;
    render();
    alert('Error generating follow-up: ' + e.message);
  }
}

async function startReview() {
  try {
    state.processing = true;
    state.view = 'dashboard';
    render();
    const review = await api('/api/review_quiz', { method: 'POST' });
    if (review.message) {
      alert(review.message);
      state.processing = false;
      render();
      return;
    }
    state.currentQuiz = review;
    state.currentAnswers = {};
    state.hintsRevealed = {};
    state.processing = false;
    state.view = 'quiz';
    render();
    window.scrollTo(0, 0);
  } catch(e) {
    state.processing = false;
    render();
    alert('Error starting review: ' + e.message);
  }
}

function viewVideoQuiz(videoId) {
  const quiz = state.quizzes.find(q => q.video_id === videoId && q.quiz_type !== 'followup' && q.quiz_type !== 'review');
  if (quiz) {
    startQuiz(quiz.id);
  }
}

function startQuiz(quizId) {
  const quiz = state.quizzes.find(q => q.id === quizId);
  if (quiz) {
    state.currentQuiz = quiz;
    state.currentAnswers = {};
    state.hintsRevealed = {};
    state.view = 'quiz';
    render();
    window.scrollTo(0, 0);
  }
}

function toggleDetail(videoId) {
  const el = document.getElementById('detail-' + videoId);
  if (el) el.classList.toggle('hidden');
}

async function saveApiKey() {
  const key = document.getElementById('apiKeyInput')?.value?.trim();
  if (!key) return;
  const msgEl = document.getElementById('settingsMsg');
  try {
    await api('/api/config', { method: 'POST', body: { api_key: key } });
    state.hasApiKey = true;
    if (msgEl) msgEl.innerHTML = '<span style="color: var(--green);">API key saved successfully.</span>';
  } catch(e) {
    if (msgEl) msgEl.innerHTML = '<span style="color: var(--red);">Error: ' + esc(e.message) + '</span>';
  }
}

// Start
init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    host = os.environ.get("LEARNENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("LEARNENGINE_PORT", "5050"))
    debug = os.environ.get("LEARNENGINE_DEBUG", "false").lower() == "true"
    print(f"\n  LearnEngine starting...")
    print(f"  Open http://{host}:{port} in your browser\n")
    app.run(host=host, port=port, debug=debug)
