"""
Adaptive Quiz Engine
Implements spaced repetition (SM-2 algorithm variant), knowledge gap tracking,
and progressive difficulty adjustment.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)


def load_json(filename: str, default=None):
    ensure_data_dir()
    path = DATA_DIR / filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(filename: str, data):
    ensure_data_dir()
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# --- Learner Profile ---

def get_learner_profile() -> dict:
    """Load or initialize the learner profile."""
    default = {
        "created_at": datetime.now().isoformat(),
        "total_quizzes_taken": 0,
        "total_questions_answered": 0,
        "overall_accuracy": 0.0,
        "topics": {},  # topic -> {strength, exposure_count, last_seen}
        "concepts": {},  # concept_id -> SM-2 params
        "weak_areas": [],
        "strong_areas": [],
        "history": [],  # recent quiz results
    }
    return load_json("learner_profile.json", default)


def save_learner_profile(profile: dict):
    save_json("learner_profile.json", profile)


# --- SM-2 Spaced Repetition ---

def sm2_update(concept: dict, quality: int) -> dict:
    """
    Update SM-2 parameters for a concept.
    quality: 0-5 (0=complete blackout, 5=perfect recall)
    Returns updated concept dict with next_review date.
    """
    if "easiness" not in concept:
        concept["easiness"] = 2.5
        concept["interval"] = 1
        concept["repetitions"] = 0

    if quality >= 3:  # correct response
        if concept["repetitions"] == 0:
            concept["interval"] = 1
        elif concept["repetitions"] == 1:
            concept["interval"] = 6
        else:
            concept["interval"] = round(concept["interval"] * concept["easiness"])
        concept["repetitions"] += 1
    else:  # incorrect
        concept["repetitions"] = 0
        concept["interval"] = 1

    # Update easiness factor
    concept["easiness"] = max(1.3,
        concept["easiness"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )

    concept["last_reviewed"] = datetime.now().isoformat()
    concept["next_review"] = (datetime.now() + timedelta(days=concept["interval"])).isoformat()
    concept["quality_history"] = concept.get("quality_history", []) + [quality]

    return concept


def get_due_concepts(profile: dict) -> list:
    """Get all concepts due for review based on spaced repetition schedule."""
    now = datetime.now()
    due = []
    for concept_id, concept in profile.get("concepts", {}).items():
        next_review = concept.get("next_review")
        if next_review:
            review_date = datetime.fromisoformat(next_review)
            if review_date <= now:
                due.append((concept_id, concept))
        else:
            due.append((concept_id, concept))
    # Sort by overdue-ness (most overdue first)
    due.sort(key=lambda x: x[1].get("next_review", "2000-01-01"))
    return due


# --- Quiz Management ---

def save_video_data(video_data: dict):
    """Save processed video data."""
    videos = load_json("videos.json", [])
    # Check if already exists
    existing = next((v for v in videos if v["video_id"] == video_data["video_id"]), None)
    if existing:
        existing.update(video_data)
    else:
        videos.append(video_data)
    save_json("videos.json", videos)


def get_all_videos() -> list:
    return load_json("videos.json", [])


def save_quiz(quiz: dict):
    """Save a generated quiz."""
    quizzes = load_json("quizzes.json", [])
    quiz["id"] = quiz.get("id", str(uuid.uuid4())[:8])
    quiz["created_at"] = quiz.get("created_at", datetime.now().isoformat())
    quizzes.append(quiz)
    save_json("quizzes.json", quizzes)
    return quiz["id"]


def get_quiz(quiz_id: str) -> dict:
    quizzes = load_json("quizzes.json", [])
    return next((q for q in quizzes if q["id"] == quiz_id), None)


def get_all_quizzes() -> list:
    return load_json("quizzes.json", [])


def record_quiz_result(quiz_id: str, answers: dict, score: float, details: list):
    """
    Record quiz results and update learner profile.
    answers: {question_index: selected_answer}
    details: [{question, correct, selected, concept_id, quality}]
    """
    profile = get_learner_profile()

    # Update overall stats
    profile["total_quizzes_taken"] += 1
    profile["total_questions_answered"] += len(answers)

    # Running average accuracy
    total_q = profile["total_questions_answered"]
    correct_q = sum(1 for d in details if d["correct"])
    if total_q > 0:
        old_total = total_q - len(answers)
        old_correct = round(profile["overall_accuracy"] * old_total)
        profile["overall_accuracy"] = (old_correct + correct_q) / total_q

    # Update SM-2 for each concept
    for detail in details:
        concept_id = detail.get("concept_id", detail.get("question", "")[:50])
        quality = 5 if detail["correct"] else 1  # simplified: correct=5, wrong=1
        if concept_id not in profile["concepts"]:
            profile["concepts"][concept_id] = {
                "topic": detail.get("topic", "general"),
                "description": detail.get("question", "")[:100],
            }
        profile["concepts"][concept_id] = sm2_update(
            profile["concepts"][concept_id], quality
        )

    # Update topic strengths
    topic_scores = {}
    for detail in details:
        topic = detail.get("topic", "general")
        if topic not in topic_scores:
            topic_scores[topic] = {"correct": 0, "total": 0}
        topic_scores[topic]["total"] += 1
        if detail["correct"]:
            topic_scores[topic]["correct"] += 1

    for topic, scores in topic_scores.items():
        if topic not in profile["topics"]:
            profile["topics"][topic] = {
                "strength": 0.5,
                "exposure_count": 0,
                "last_seen": None
            }
        t = profile["topics"][topic]
        new_strength = scores["correct"] / scores["total"]
        # Exponential moving average
        t["strength"] = 0.7 * t["strength"] + 0.3 * new_strength
        t["exposure_count"] += scores["total"]
        t["last_seen"] = datetime.now().isoformat()

    # Identify weak and strong areas
    profile["weak_areas"] = [
        t for t, data in profile["topics"].items()
        if data["strength"] < 0.6 and data["exposure_count"] >= 3
    ]
    profile["strong_areas"] = [
        t for t, data in profile["topics"].items()
        if data["strength"] >= 0.85 and data["exposure_count"] >= 3
    ]

    # Record in history
    profile["history"].append({
        "quiz_id": quiz_id,
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "num_questions": len(answers),
        "correct": correct_q,
    })

    # Keep last 100 history entries
    profile["history"] = profile["history"][-100:]

    save_learner_profile(profile)

    return {
        "score": score,
        "correct": correct_q,
        "total": len(answers),
        "weak_areas": profile["weak_areas"],
        "due_for_review": len(get_due_concepts(profile)),
    }


def get_learning_stats() -> dict:
    """Get summary learning statistics."""
    profile = get_learner_profile()
    videos = get_all_videos()
    quizzes = get_all_quizzes()
    due = get_due_concepts(profile)

    return {
        "total_videos": len(videos),
        "total_quizzes": len(quizzes),
        "total_questions_answered": profile["total_questions_answered"],
        "overall_accuracy": round(profile["overall_accuracy"] * 100, 1),
        "weak_areas": profile["weak_areas"],
        "strong_areas": profile["strong_areas"],
        "concepts_due_for_review": len(due),
        "topics": profile["topics"],
        "recent_history": profile["history"][-10:],
    }
