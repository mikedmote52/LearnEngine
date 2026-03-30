"""
Database Layer - SQLite backend for LearnEngine
Replaces JSON file storage with proper concurrent-safe persistence.
Handles migrations from existing JSON data automatically.
"""

import json
import sqlite3
import uuid
import os
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
# Allow overriding DB path via environment for filesystems that don't support SQLite locking (e.g., FUSE)
_db_dir = Path(os.environ.get("LEARNENGINE_DB_DIR", str(DATA_DIR)))
DB_PATH = _db_dir / "learnengine.db"


def get_db():
    """Get a database connection."""
    _db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    # Try WAL mode for better concurrency; fall back gracefully
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            transcript_json TEXT,
            analysis_json TEXT,
            processed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS quizzes (
            id TEXT PRIMARY KEY,
            video_id TEXT,
            video_title TEXT,
            quiz_type TEXT DEFAULT 'initial',
            parent_quiz_id TEXT,
            focus_areas_json TEXT,
            questions_json TEXT,
            difficulty_level TEXT,
            created_at TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        );

        CREATE TABLE IF NOT EXISTS learner_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            created_at TEXT,
            total_quizzes_taken INTEGER DEFAULT 0,
            total_questions_answered INTEGER DEFAULT 0,
            overall_accuracy REAL DEFAULT 0.0,
            topics_json TEXT DEFAULT '{}',
            concepts_json TEXT DEFAULT '{}',
            weak_areas_json TEXT DEFAULT '[]',
            strong_areas_json TEXT DEFAULT '[]',
            learning_style_json TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id TEXT,
            timestamp TEXT,
            score REAL,
            num_questions INTEGER,
            correct INTEGER,
            details_json TEXT,
            FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
        );

        CREATE TABLE IF NOT EXISTS concept_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id TEXT,
            quiz_id TEXT,
            question TEXT,
            correct INTEGER,
            selected_answer TEXT,
            correct_answer TEXT,
            quality INTEGER,
            timestamp TEXT,
            explanation_viewed INTEGER DEFAULT 0
        );
    """)
    conn.commit()

    # Ensure learner profile row exists
    row = conn.execute("SELECT id FROM learner_profile WHERE id=1").fetchone()
    if not row:
        conn.execute(
            "INSERT INTO learner_profile (id, created_at) VALUES (1, ?)",
            (datetime.now().isoformat(),)
        )
        conn.commit()

    conn.close()
    _migrate_from_json()


def _migrate_from_json():
    """One-time migration from old JSON files if they exist."""
    videos_path = DATA_DIR / "videos.json"
    quizzes_path = DATA_DIR / "quizzes.json"
    profile_path = DATA_DIR / "learner_profile.json"

    conn = get_db()

    if videos_path.exists():
        try:
            with open(videos_path) as f:
                videos = json.load(f)
            for v in videos:
                existing = conn.execute("SELECT video_id FROM videos WHERE video_id=?", (v["video_id"],)).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO videos (video_id, url, title, transcript_json, analysis_json, processed_at) VALUES (?,?,?,?,?,?)",
                        (v["video_id"], v.get("url",""), v.get("title",""),
                         json.dumps(v.get("transcript",{})), json.dumps(v.get("analysis",{})),
                         v.get("processed_at",""))
                    )
            conn.commit()
            videos_path.rename(videos_path.with_suffix(".json.migrated"))
        except Exception as e:
            print(f"Migration warning (videos): {e}")

    if quizzes_path.exists():
        try:
            with open(quizzes_path) as f:
                quizzes = json.load(f)
            for q in quizzes:
                existing = conn.execute("SELECT id FROM quizzes WHERE id=?", (q.get("id",""),)).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO quizzes (id, video_id, video_title, quiz_type, parent_quiz_id, focus_areas_json, questions_json, difficulty_level, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        (q.get("id", str(uuid.uuid4())[:8]), q.get("video_id",""), q.get("video_title",""),
                         q.get("quiz_type","initial"), q.get("parent_quiz_id"),
                         json.dumps(q.get("focus_areas",[])), json.dumps(q.get("questions",[])),
                         q.get("difficulty_level","mixed"), q.get("created_at", datetime.now().isoformat()))
                    )
            conn.commit()
            quizzes_path.rename(quizzes_path.with_suffix(".json.migrated"))
        except Exception as e:
            print(f"Migration warning (quizzes): {e}")

    if profile_path.exists():
        try:
            with open(profile_path) as f:
                p = json.load(f)
            conn.execute("""
                UPDATE learner_profile SET
                    total_quizzes_taken=?, total_questions_answered=?, overall_accuracy=?,
                    topics_json=?, concepts_json=?, weak_areas_json=?, strong_areas_json=?
                WHERE id=1
            """, (
                p.get("total_quizzes_taken",0), p.get("total_questions_answered",0),
                p.get("overall_accuracy",0.0), json.dumps(p.get("topics",{})),
                json.dumps(p.get("concepts",{})), json.dumps(p.get("weak_areas",[])),
                json.dumps(p.get("strong_areas",[]))
            ))
            conn.commit()
            # Migrate history
            for h in p.get("history", []):
                conn.execute(
                    "INSERT INTO quiz_history (quiz_id, timestamp, score, num_questions, correct) VALUES (?,?,?,?,?)",
                    (h.get("quiz_id",""), h.get("timestamp",""), h.get("score",0),
                     h.get("num_questions",0), h.get("correct",0))
                )
            conn.commit()
            profile_path.rename(profile_path.with_suffix(".json.migrated"))
        except Exception as e:
            print(f"Migration warning (profile): {e}")

    conn.close()


# ---- Video Operations ----

def save_video(video_data: dict):
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO videos (video_id, url, title, transcript_json, analysis_json, processed_at)
        VALUES (?,?,?,?,?,?)
    """, (
        video_data["video_id"], video_data.get("url",""), video_data.get("title",""),
        json.dumps(video_data.get("transcript",{})), json.dumps(video_data.get("analysis",{})),
        video_data.get("processed_at", datetime.now().isoformat())
    ))
    conn.commit()
    conn.close()


def get_all_videos() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM videos ORDER BY processed_at DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "video_id": r["video_id"], "url": r["url"], "title": r["title"],
            "transcript": json.loads(r["transcript_json"]),
            "analysis": json.loads(r["analysis_json"]),
            "processed_at": r["processed_at"],
        })
    return result


def get_video(video_id: str) -> dict:
    conn = get_db()
    r = conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
    conn.close()
    if not r:
        return None
    return {
        "video_id": r["video_id"], "url": r["url"], "title": r["title"],
        "transcript": json.loads(r["transcript_json"]),
        "analysis": json.loads(r["analysis_json"]),
        "processed_at": r["processed_at"],
    }


# ---- Quiz Operations ----

def save_quiz(quiz_data: dict) -> str:
    qid = quiz_data.get("id", str(uuid.uuid4())[:8])
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO quizzes (id, video_id, video_title, quiz_type, parent_quiz_id,
            focus_areas_json, questions_json, difficulty_level, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        qid, quiz_data.get("video_id",""), quiz_data.get("video_title",""),
        quiz_data.get("quiz_type","initial"), quiz_data.get("parent_quiz_id"),
        json.dumps(quiz_data.get("focus_areas",[])), json.dumps(quiz_data.get("questions",[])),
        quiz_data.get("difficulty_level","mixed"),
        quiz_data.get("created_at", datetime.now().isoformat())
    ))
    conn.commit()
    conn.close()
    return qid


def get_quiz(quiz_id: str) -> dict:
    conn = get_db()
    r = conn.execute("SELECT * FROM quizzes WHERE id=?", (quiz_id,)).fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r["id"], "video_id": r["video_id"], "video_title": r["video_title"],
        "quiz_type": r["quiz_type"], "parent_quiz_id": r["parent_quiz_id"],
        "focus_areas": json.loads(r["focus_areas_json"] or "[]"),
        "questions": json.loads(r["questions_json"] or "[]"),
        "difficulty_level": r["difficulty_level"],
        "created_at": r["created_at"],
    }


def get_all_quizzes() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM quizzes ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{
        "id": r["id"], "video_id": r["video_id"], "video_title": r["video_title"],
        "quiz_type": r["quiz_type"], "parent_quiz_id": r["parent_quiz_id"],
        "focus_areas": json.loads(r["focus_areas_json"] or "[]"),
        "questions": json.loads(r["questions_json"] or "[]"),
        "difficulty_level": r["difficulty_level"],
        "created_at": r["created_at"],
    } for r in rows]


# ---- Learner Profile & SM-2 ----

def get_learner_profile() -> dict:
    conn = get_db()
    r = conn.execute("SELECT * FROM learner_profile WHERE id=1").fetchone()
    conn.close()
    return {
        "total_quizzes_taken": r["total_quizzes_taken"],
        "total_questions_answered": r["total_questions_answered"],
        "overall_accuracy": r["overall_accuracy"],
        "topics": json.loads(r["topics_json"] or "{}"),
        "concepts": json.loads(r["concepts_json"] or "{}"),
        "weak_areas": json.loads(r["weak_areas_json"] or "[]"),
        "strong_areas": json.loads(r["strong_areas_json"] or "[]"),
        "learning_style": json.loads(r["learning_style_json"] or "{}"),
    }


def save_learner_profile(profile: dict):
    conn = get_db()
    conn.execute("""
        UPDATE learner_profile SET
            total_quizzes_taken=?, total_questions_answered=?, overall_accuracy=?,
            topics_json=?, concepts_json=?, weak_areas_json=?, strong_areas_json=?,
            learning_style_json=?
        WHERE id=1
    """, (
        profile["total_quizzes_taken"], profile["total_questions_answered"],
        profile["overall_accuracy"], json.dumps(profile.get("topics",{})),
        json.dumps(profile.get("concepts",{})), json.dumps(profile.get("weak_areas",[])),
        json.dumps(profile.get("strong_areas",[])), json.dumps(profile.get("learning_style",{}))
    ))
    conn.commit()
    conn.close()


def sm2_update(concept: dict, quality: int) -> dict:
    """
    SM-2 spaced repetition with proper 0-5 quality scale.
    0 = complete blackout, 1 = wrong but recognized after seeing answer,
    2 = wrong but close, 3 = correct with difficulty, 4 = correct with hesitation,
    5 = perfect instant recall.
    """
    if "easiness" not in concept:
        concept["easiness"] = 2.5
        concept["interval"] = 1
        concept["repetitions"] = 0

    if quality >= 3:
        if concept["repetitions"] == 0:
            concept["interval"] = 1
        elif concept["repetitions"] == 1:
            concept["interval"] = 6
        else:
            concept["interval"] = round(concept["interval"] * concept["easiness"])
        concept["repetitions"] += 1
    else:
        concept["repetitions"] = 0
        concept["interval"] = 1

    concept["easiness"] = max(1.3,
        concept["easiness"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )

    concept["last_reviewed"] = datetime.now().isoformat()
    concept["next_review"] = (datetime.now() + timedelta(days=concept["interval"])).isoformat()
    concept["quality_history"] = concept.get("quality_history", []) + [quality]
    concept["times_tested"] = concept.get("times_tested", 0) + 1
    concept["times_correct"] = concept.get("times_correct", 0) + (1 if quality >= 3 else 0)
    concept["mastery"] = concept["times_correct"] / concept["times_tested"] if concept["times_tested"] > 0 else 0

    return concept


def get_due_concepts(profile: dict) -> list:
    now = datetime.now()
    due = []
    for concept_id, concept in profile.get("concepts", {}).items():
        next_review = concept.get("next_review")
        if next_review:
            try:
                review_date = datetime.fromisoformat(next_review)
                if review_date <= now:
                    due.append((concept_id, concept))
            except (ValueError, TypeError):
                due.append((concept_id, concept))
        else:
            due.append((concept_id, concept))
    due.sort(key=lambda x: x[1].get("next_review", "2000-01-01"))
    return due


def record_quiz_result(quiz_id: str, answers: dict, score: float, details: list) -> dict:
    """Record results and update profile with granular quality scoring."""
    profile = get_learner_profile()

    profile["total_quizzes_taken"] += 1
    profile["total_questions_answered"] += len(answers)

    correct_q = sum(1 for d in details if d["correct"])
    total_q = profile["total_questions_answered"]
    if total_q > 0:
        old_total = total_q - len(answers)
        old_correct = round(profile["overall_accuracy"] * old_total)
        profile["overall_accuracy"] = (old_correct + correct_q) / total_q

    # Update SM-2 with granular quality
    conn = get_db()
    for detail in details:
        concept_id = detail.get("concept_id", detail.get("question", "")[:50])

        # Granular quality scoring
        if detail["correct"]:
            time_factor = detail.get("time_taken", 0)
            if time_factor and time_factor < 10:
                quality = 5  # fast and correct
            elif time_factor and time_factor < 30:
                quality = 4  # correct with some thought
            else:
                quality = 3  # correct but slow / no timing data
        else:
            # Check if answer was close (same topic, adjacent concept)
            quality = 1  # wrong

        if concept_id not in profile["concepts"]:
            profile["concepts"][concept_id] = {
                "topic": detail.get("topic", "general"),
                "description": detail.get("question", "")[:200],
                "name": detail.get("concept_name", concept_id),
            }
        profile["concepts"][concept_id] = sm2_update(
            profile["concepts"][concept_id], quality
        )

        # Record individual attempt
        conn.execute("""
            INSERT INTO concept_attempts (concept_id, quiz_id, question, correct, selected_answer,
                correct_answer, quality, timestamp)
            VALUES (?,?,?,?,?,?,?,?)
        """, (concept_id, quiz_id, detail.get("question",""), 1 if detail["correct"] else 0,
              detail.get("selected",""), detail.get("correct_answer",""),
              quality, datetime.now().isoformat()))

    # Update topic strengths with EMA
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
                "strength": 0.5, "exposure_count": 0, "last_seen": None
            }
        t = profile["topics"][topic]
        new_strength = scores["correct"] / scores["total"]
        t["strength"] = 0.7 * t["strength"] + 0.3 * new_strength
        t["exposure_count"] += scores["total"]
        t["last_seen"] = datetime.now().isoformat()

    # Identify weak/strong areas
    profile["weak_areas"] = [
        t for t, data in profile["topics"].items()
        if data["strength"] < 0.6 and data["exposure_count"] >= 2
    ]
    profile["strong_areas"] = [
        t for t, data in profile["topics"].items()
        if data["strength"] >= 0.85 and data["exposure_count"] >= 2
    ]

    # Detect learning style patterns
    _update_learning_style(profile, details)

    # Record history
    conn.execute("""
        INSERT INTO quiz_history (quiz_id, timestamp, score, num_questions, correct, details_json)
        VALUES (?,?,?,?,?,?)
    """, (quiz_id, datetime.now().isoformat(), score, len(answers), correct_q, json.dumps(details)))
    conn.commit()
    conn.close()

    save_learner_profile(profile)

    return {
        "score": score,
        "correct": correct_q,
        "total": len(answers),
        "weak_areas": profile["weak_areas"],
        "strong_areas": profile["strong_areas"],
        "due_for_review": len(get_due_concepts(profile)),
        "learning_style": profile.get("learning_style", {}),
    }


def _update_learning_style(profile: dict, details: list):
    """Track patterns in what the learner gets right/wrong to adapt teaching approach."""
    style = profile.get("learning_style", {})

    # Track which difficulty levels the learner succeeds at
    diff_results = style.get("difficulty_results", {"easy": [0,0], "medium": [0,0], "hard": [0,0]})
    for d in details:
        diff = d.get("difficulty", "medium")
        if diff in diff_results:
            diff_results[diff][1] += 1  # total
            if d["correct"]:
                diff_results[diff][0] += 1  # correct

    style["difficulty_results"] = diff_results

    # Determine optimal difficulty
    rates = {}
    for diff, (c, t) in diff_results.items():
        if t >= 3:
            rates[diff] = c / t
    style["difficulty_rates"] = rates

    # Track concept types that are harder for this learner
    wrong_topics = [d.get("topic","") for d in details if not d["correct"]]
    right_topics = [d.get("topic","") for d in details if d["correct"]]
    style["recent_struggles"] = wrong_topics
    style["recent_strengths"] = right_topics

    # Determine if learner needs more examples, more theory, or more application
    total_wrong = len(wrong_topics)
    total_right = len(right_topics)
    total = total_wrong + total_right
    if total >= 5:
        accuracy = total_right / total
        if accuracy < 0.4:
            style["teaching_mode"] = "foundational"  # needs basics re-explained
        elif accuracy < 0.7:
            style["teaching_mode"] = "scaffolded"  # needs step-by-step building
        else:
            style["teaching_mode"] = "challenging"  # ready for harder material

    profile["learning_style"] = style


def get_learning_stats() -> dict:
    profile = get_learner_profile()
    conn = get_db()
    video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    quiz_count = conn.execute("SELECT COUNT(*) FROM quizzes").fetchone()[0]
    recent = conn.execute(
        "SELECT * FROM quiz_history ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()
    conn.close()

    due = get_due_concepts(profile)

    # Concept mastery breakdown
    mastered = 0
    learning = 0
    struggling = 0
    for cid, c in profile.get("concepts", {}).items():
        mastery = c.get("mastery", 0)
        if mastery >= 0.85:
            mastered += 1
        elif mastery >= 0.5:
            learning += 1
        else:
            struggling += 1

    return {
        "total_videos": video_count,
        "total_quizzes": quiz_count,
        "total_questions_answered": profile["total_questions_answered"],
        "overall_accuracy": round(profile["overall_accuracy"] * 100, 1),
        "weak_areas": profile["weak_areas"],
        "strong_areas": profile["strong_areas"],
        "concepts_due_for_review": len(due),
        "concepts_mastered": mastered,
        "concepts_learning": learning,
        "concepts_struggling": struggling,
        "topics": profile["topics"],
        "learning_style": profile.get("learning_style", {}),
        "recent_history": [
            {"quiz_id": r["quiz_id"], "timestamp": r["timestamp"],
             "score": r["score"], "num_questions": r["num_questions"],
             "correct": r["correct"]}
            for r in recent
        ],
    }


def get_concept_history(concept_id: str) -> list:
    """Get full attempt history for a specific concept."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM concept_attempts WHERE concept_id=? ORDER BY timestamp DESC",
        (concept_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
