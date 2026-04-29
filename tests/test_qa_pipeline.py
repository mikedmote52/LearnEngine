"""Stub tests for the QA gate + judge pipeline.

These tests monkeypatch _openrouter_chat so we can exercise the full filter
logic deterministically without burning real API credits or hitting Vercel.
The point is to prove three things:

  1. Adversarial transcript: lies that Pass 1 flagged + lies that the judge
     can score should both be dropped.
  2. Blank-options bug (the Q4-C / Q5-B screenshot): _sanitize_quiz_list now
     drops the entire question if any option text is empty.
  3. Clean transcript: a normal payload survives the gate + judge with a
     reasonable count.

Run from repo root:
  python tests/test_qa_pipeline.py
"""

import json
import os
import sys
import unittest
from unittest import mock

# Force STRICT_QA on for tests (the env var defaults to true anyway, but be
# explicit so we never get a confusing "all questions kept because QA is off").
os.environ["STRICT_QA"] = "true"
os.environ["STRICT_QA_THRESHOLD"] = "0.7"
os.environ.setdefault("OPENROUTER_API_KEY", "stub-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api import index as le  # noqa: E402


def _mk_q(stem, options, **extra):
    """Helper: build a quiz question dict with the standard 4-option shape.
    options is a list of (text, correct) tuples."""
    q = {
        "question": stem,
        "options": [
            {
                "label": chr(65 + i),
                "text": text,
                "correct": correct,
                "why_wrong": None if correct else "wrong because.",
            }
            for i, (text, correct) in enumerate(options)
        ],
    }
    q.update(extra)
    return q


# ---- Stub for _openrouter_chat ----
# Returns canned strings depending on the model + prompt content. The pipeline
# only ever calls _openrouter_chat through call_llm (the wrapper), so this
# replaces both Haiku gate calls and Sonnet judge calls.

def make_stub(gate_drops=None, judge_scores=None, judge_fails=False):
    """Build an _openrouter_chat replacement.
      gate_drops: list of {"q":idx, "c":idx, "reason":str} the gate should return
      judge_scores: list of per-question score dicts (key "q" -> idx)
      judge_fails: if True, judge call raises (simulating outage)
    """
    gate_drops = gate_drops or []
    judge_scores = judge_scores or []

    def _stub(model, messages, max_tokens=4000, stream=False):
        prompt_text = ""
        if messages and isinstance(messages[0].get("content"), list):
            for blk in messages[0]["content"]:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    prompt_text += blk.get("text", "")
        elif messages and isinstance(messages[0].get("content"), str):
            prompt_text = messages[0]["content"]

        if "FLAGGED CLAIMS" in prompt_text:
            return json.dumps({"drops": gate_drops})
        if "scored against the SOURCE" in prompt_text or "QUESTIONS:" in prompt_text:
            if judge_fails:
                raise RuntimeError("simulated judge outage")
            return json.dumps({"scores": judge_scores})
        # Anything else — return empty JSON so any unexpected callers don't
        # blow up the test.
        return "{}"

    return _stub


class QASanitizeBlankOptionsTest(unittest.TestCase):
    """Bug from Mike's 20-question screenshot: Q5-B rendered as a blank choice
    because the option's text was empty. The renderer didn't notice and the
    user had to guess. Sanitize layer must drop the whole question."""

    def test_drops_question_with_empty_option_text(self):
        quiz = [
            _mk_q("Real question", [
                ("Option A", False),
                ("Option B", True),
                ("Option C", False),
                ("Option D", False),
            ]),
            _mk_q("Question with blank Q5-B", [
                ("Plausible distractor", False),
                ("", True),  # the bug
                ("Another distractor", False),
                ("Yet another", False),
            ]),
            _mk_q("Question with whitespace-only option", [
                ("Real text", False),
                ("   \t  ", False),
                ("Real correct text", True),
                ("Real distractor", False),
            ]),
        ]
        cleaned = le._sanitize_quiz_list(quiz)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["question"], "Real question")


class QAGateAdversarialTest(unittest.TestCase):
    """Feed an adversarial Pass-1/Pass-2 setup. Pass 1 flagged misinformation;
    Pass 2 went ahead and asked questions whose 'correct' answer is the
    flagged claim. The gate should drop those."""

    def test_drops_questions_matching_misinformation_flags(self):
        analysis = {
            "summary": "A talk about basic physics and chemistry.",
            "fact_check": [
                {"claim": "Water boils at 50 degrees Celsius at sea level.",
                 "assessment": "inaccurate",
                 "correction": "It boils at 100°C at sea level."},
                {"claim": "Mount Everest is in Africa.",
                 "assessment": "inaccurate",
                 "correction": "Everest is on the Nepal/Tibet border."},
            ],
            "misinformation_flags": [
                "The Sun orbits the Earth",
            ],
        }
        quiz = [
            _mk_q("At sea level, water boils at what temperature?", [
                ("100°C", False),
                ("50°C", True),  # the lie, marked correct — must be dropped
                ("200°C", False),
                ("0°C", False),
            ]),
            _mk_q("Which planet does the Sun orbit?", [
                ("Earth", True),  # also flagged
                ("None — Sun is the center of the solar system", False),
                ("Jupiter", False),
                ("Mars", False),
            ]),
            _mk_q("On which continent is Mount Everest located?", [
                ("Asia", False),
                ("Africa", True),  # flagged
                ("Europe", False),
                ("South America", False),
            ]),
            _mk_q("What is the chemical formula for water?", [
                ("H2O", True),  # legitimate question, should pass
                ("CO2", False),
                ("NaCl", False),
                ("CH4", False),
            ]),
        ]

        # Stub: gate identifies the three lies (q indices 0, 1, 2)
        gate_drops = [
            {"q": 0, "c": 0, "reason": "matches flagged claim about boiling"},
            {"q": 1, "c": 0, "reason": "matches Sun orbits Earth flag"},
            {"q": 2, "c": 1, "reason": "matches Everest in Africa flag"},
        ]
        # Stub: judge passes the only remaining (legitimate H2O) question
        judge_scores = [
            {"q": 0, "correct_supported": True, "distractors_wrong": True,
             "single_answer": True, "self_contained": True, "confidence": 0.95},
        ]

        with mock.patch.object(le, "_openrouter_chat",
                               side_effect=make_stub(gate_drops, judge_scores)):
            filtered, qa_meta = le._apply_qa_pipeline(quiz, analysis=analysis,
                                                     transcript="A talk.")

        self.assertEqual(qa_meta["pre_filter_count"], 4)
        self.assertEqual(qa_meta["gate_dropped"], 3)
        self.assertEqual(qa_meta["final_count"], 1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["question"],
                         "What is the chemical formula for water?")
        # Confirm the question whose correct answer was a flagged claim is
        # absent — this is the load-bearing assertion.
        kept_correct_texts = [le._correct_option_text(q) for q in filtered]
        for lie in ("50°C", "Earth", "Africa"):
            self.assertNotIn(lie, kept_correct_texts)


class QAJudgeFiltersLowConfidenceTest(unittest.TestCase):
    """Even when the gate finds nothing to flag, the judge should drop
    questions where the correct answer isn't supported, distractors aren't
    clearly wrong, or confidence is below threshold."""

    def test_drops_below_threshold_and_failing_bools(self):
        analysis = {
            "summary": "A clean cell biology talk.",
            "fact_check": [],
            "misinformation_flags": [],
        }
        quiz = [
            _mk_q("Mitochondria are known as the what of the cell?", [
                ("Powerhouse", True), ("Brain", False),
                ("Skeleton", False), ("Skin", False),
            ]),
            _mk_q("Ambiguous question with two defensible answers.", [
                ("A", True), ("B", False), ("C", False), ("D", False),
            ]),
            _mk_q("Low confidence — judge isn't sure.", [
                ("Maybe", True), ("Maybe not", False),
                ("Could be", False), ("Unclear", False),
            ]),
        ]
        judge_scores = [
            {"q": 0, "correct_supported": True, "distractors_wrong": True,
             "single_answer": True, "self_contained": True, "confidence": 0.92},
            {"q": 1, "correct_supported": True, "distractors_wrong": True,
             "single_answer": False, "self_contained": True, "confidence": 0.85,
             "reason": "ambiguous"},
            {"q": 2, "correct_supported": True, "distractors_wrong": True,
             "single_answer": True, "self_contained": True, "confidence": 0.45},
        ]
        with mock.patch.object(le, "_openrouter_chat",
                               side_effect=make_stub([], judge_scores)):
            filtered, qa_meta = le._apply_qa_pipeline(quiz, analysis=analysis,
                                                     transcript="cell biology talk")
        self.assertEqual(qa_meta["gate_dropped"], 0)
        self.assertEqual(qa_meta["judge_dropped"], 2)
        self.assertEqual(qa_meta["final_count"], 1)
        self.assertEqual(filtered[0]["question"],
                         "Mitochondria are known as the what of the cell?")


class QACleanTranscriptTest(unittest.TestCase):
    """Sanity check: when nothing is wrong, the pipeline keeps the questions."""

    def test_keeps_clean_questions(self):
        analysis = {"summary": "clean", "fact_check": [], "misinformation_flags": []}
        quiz = [
            _mk_q(f"Clean question {i}?", [
                (f"Right{i}", True),
                (f"Wrong{i}A", False),
                (f"Wrong{i}B", False),
                (f"Wrong{i}C", False),
            ])
            for i in range(10)
        ]
        # All-passing judge scores
        judge_scores = [
            {"q": i, "correct_supported": True, "distractors_wrong": True,
             "single_answer": True, "self_contained": True, "confidence": 0.9}
            for i in range(10)
        ]
        with mock.patch.object(le, "_openrouter_chat",
                               side_effect=make_stub([], judge_scores)):
            filtered, qa_meta = le._apply_qa_pipeline(quiz, analysis=analysis,
                                                     transcript="clean transcript")
        self.assertEqual(qa_meta["final_count"], 10)
        self.assertGreaterEqual(len(filtered), 7)


class QAJudgeFailOpenTest(unittest.TestCase):
    """If the judge call itself errors (network blip, model outage), we should
    fail open: keep the questions and surface the failure in drop_reasons."""

    def test_judge_outage_keeps_questions(self):
        analysis = {"summary": "x", "fact_check": [], "misinformation_flags": []}
        quiz = [
            _mk_q("Q1?", [("A", True), ("B", False), ("C", False), ("D", False)]),
            _mk_q("Q2?", [("A", False), ("B", True), ("C", False), ("D", False)]),
        ]
        with mock.patch.object(le, "_openrouter_chat",
                               side_effect=make_stub([], [], judge_fails=True)):
            filtered, qa_meta = le._apply_qa_pipeline(quiz, analysis=analysis)
        self.assertEqual(len(filtered), 2)
        # Failure should be surfaced rather than swallowed.
        self.assertTrue(any("judge" in r.get("reason", "")
                            or "judge call failed" in r.get("question", "")
                            for r in qa_meta["drop_reasons"]))


class QAStrictDisabledTest(unittest.TestCase):
    """STRICT_QA=false should pass everything through with qa_meta still
    populated (just with enabled=False)."""

    def test_disabled_flag_passes_through(self):
        try:
            os.environ["STRICT_QA"] = "false"
            quiz = [_mk_q("Q?", [("A", True), ("B", False), ("C", False), ("D", False)])]
            filtered, qa_meta = le._apply_qa_pipeline(quiz, analysis={})
            self.assertEqual(qa_meta["enabled"], False)
            self.assertEqual(len(filtered), 1)
        finally:
            os.environ["STRICT_QA"] = "true"


if __name__ == "__main__":
    unittest.main(verbosity=2)
