"""
Content Analyzer
Uses the Anthropic Claude API for transcript analysis, fact-checking,
key concept extraction, and adaptive quiz generation.

Two modes:
1. Standalone: calls Claude API directly (requires API key)
2. Cowork: generates prompts that Cowork can process natively (zero cost)
"""

import json
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MODEL = os.environ.get("LEARNENGINE_MODEL", "claude-sonnet-4-20250514")

# Try to import anthropic SDK - graceful fallback if not installed
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def get_api_key() -> str:
    """Get API key from environment or config file."""
    # Prefer environment variable
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        return env_key
    # Fall back to config file
    config_path = DATA_DIR / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("anthropic_api_key", "")
        except (json.JSONDecodeError, IOError):
            return ""
    return ""


def save_api_key(key: str):
    """Save API key to config."""
    if not key or not key.startswith("sk-"):
        raise ValueError("Invalid API key format")
    DATA_DIR.mkdir(exist_ok=True)
    config_path = DATA_DIR / "config.json"
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            config = {}
    config["anthropic_api_key"] = key
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    # Restrict file permissions (owner read/write only)
    try:
        os.chmod(config_path, 0o600)
    except OSError:
        pass


def _parse_json_response(text: str) -> dict:
    """Robustly extract JSON from Claude's response."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        # Remove first line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in the text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not extract JSON from response", text, 0)


def analyze_transcript(transcript_text: str, video_title: str = "",
                       api_key: str = None, learner_context: dict = None) -> dict:
    """
    Analyze a transcript using Claude API.
    Returns structured analysis with key concepts, fact-check results, and generated quiz.

    learner_context: optional dict with learner profile info to adapt quiz difficulty/style.
    """
    if not api_key:
        api_key = get_api_key()

    if not api_key or not HAS_ANTHROPIC:
        return _generate_analysis_prompt(transcript_text, video_title)

    if not transcript_text or len(transcript_text.strip()) < 50:
        return {"error": "Transcript is too short to analyze meaningfully."}

    client = anthropic.Anthropic(api_key=api_key)

    # Truncate very long transcripts to stay within context limits
    max_chars = 80000
    truncated = transcript_text[:max_chars]
    if len(transcript_text) > max_chars:
        truncated += "\n\n[TRANSCRIPT TRUNCATED - original was longer]"

    # Build adaptive context based on learner profile
    learner_section = ""
    if learner_context:
        style = learner_context.get("learning_style", {})
        teaching_mode = style.get("teaching_mode", "scaffolded")
        weak = learner_context.get("weak_areas", [])
        strong = learner_context.get("strong_areas", [])
        accuracy = learner_context.get("overall_accuracy", 50)

        learner_section = f"""
LEARNER PROFILE (adapt quiz to this learner):
- Overall accuracy so far: {accuracy}%
- Teaching mode: {teaching_mode}
- Weak areas: {', '.join(weak) if weak else 'None identified yet'}
- Strong areas: {', '.join(strong) if strong else 'None identified yet'}
- Difficulty preference: {"Start easier, build up" if accuracy < 60 else "Mix of difficulties" if accuracy < 80 else "Challenge them"}

ADAPTATION RULES:
- If teaching_mode is "foundational": Use simpler language in explanations, include analogies, break complex ideas into parts
- If teaching_mode is "scaffolded": Build questions in sequence where each builds on the previous concept
- If teaching_mode is "challenging": Include synthesis questions that combine multiple concepts, add "what if" scenarios
- If they have weak areas matching topics in this video: generate extra questions on those topics with richer explanations
"""

    prompt = f"""You are an expert educational content designer building an adaptive learning system. Analyze this video transcript and create materials that will help the learner achieve DEEP UNDERSTANDING - not just memorization.

VIDEO TITLE: {video_title}
{learner_section}
TRANSCRIPT:
{truncated}

Respond with a JSON object (no markdown fencing, just raw JSON) containing:

{{
  "summary": "3-5 sentence summary of the core content",
  "key_concepts": [
    {{
      "id": "concept_1",
      "name": "Short concept name",
      "explanation": "Clear explanation of this concept",
      "simple_analogy": "An everyday analogy that makes this concept intuitive",
      "topic": "Category/topic area",
      "importance": "high/medium/low",
      "prerequisites": ["concept IDs this depends on understanding first"],
      "timestamp_hint": "approximate location in video if discernible"
    }}
  ],
  "concept_map": [
    {{"from": "concept_1", "to": "concept_2", "relationship": "how they connect"}}
  ],
  "fact_check": [
    {{
      "claim": "Specific claim made in the video",
      "assessment": "accurate/partially_accurate/inaccurate/unverifiable",
      "correction": "If inaccurate, what's actually true. Null if accurate.",
      "confidence": "high/medium/low",
      "reasoning": "Why this assessment"
    }}
  ],
  "misinformation_flags": [
    {{
      "statement": "The problematic statement",
      "issue": "What's wrong with it",
      "severity": "high/medium/low"
    }}
  ],
  "bias_notes": "Any notable biases, one-sided framing, or missing context",
  "difficulty_level": "beginner/intermediate/advanced",
  "prerequisite_knowledge": ["List of things you should already know"],
  "learning_objectives": ["By the end, you should be able to..."],
  "quiz": [
    {{
      "question": "Clear, specific question testing understanding (not just recall)",
      "concept_id": "which key_concept this tests",
      "concept_name": "human-readable concept name",
      "topic": "topic area",
      "difficulty": "easy/medium/hard",
      "bloom_level": "remember/understand/apply/analyze/evaluate/create",
      "options": [
        {{"label": "A", "text": "Option text", "correct": false, "why_wrong": "Why this is incorrect"}},
        {{"label": "B", "text": "Option text", "correct": true, "why_wrong": null}},
        {{"label": "C", "text": "Option text", "correct": false, "why_wrong": "Why this is incorrect"}},
        {{"label": "D", "text": "Option text", "correct": false, "why_wrong": "Why this is incorrect"}}
      ],
      "explanation": "Why the correct answer is correct, explained clearly",
      "common_misconception": "The most common mistake people make with this concept",
      "deeper_insight": "Something beyond the video that deepens understanding",
      "hint": "A nudge toward the right answer without giving it away"
    }}
  ]
}}

CRITICAL GUIDELINES:
- Generate 10-15 quiz questions covering ALL key concepts with redundancy on important ones
- SEQUENCE questions so earlier ones scaffold understanding for later ones
- Bloom's taxonomy distribution: 20% remember, 30% understand, 25% apply, 15% analyze, 10% evaluate
- Fact-check ALL specific claims, statistics, dates, and scientific assertions
- Flag any misinformation, exaggeration, or misleading framing
- Make wrong options plausible - they should represent REAL misconceptions people have
- Each "deeper_insight" should genuinely teach something new and valuable
- The "hint" should guide thinking process, not reveal the answer
- The "common_misconception" helps the learner avoid typical errors
- Be rigorous on fact-checking - don't let vague or inflated claims pass
- learning_objectives should be concrete and measurable (not vague)"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=12000,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text.strip()
        analysis = _parse_json_response(response_text)
        analysis["_source"] = "claude_api"
        analysis["_model"] = MODEL
        return analysis

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse Claude response as JSON: {e}",
            "raw_response": response_text[:2000] if 'response_text' in locals() else None,
            "_source": "claude_api_error"
        }
    except Exception as e:
        return {
            "error": f"Claude API error: {str(e)}",
            "_source": "claude_api_error"
        }


def generate_followup_quiz(
    weak_concepts: list,
    previous_wrong_answers: list,
    transcript_text: str,
    learner_context: dict = None,
    api_key: str = None
) -> dict:
    """
    Generate a targeted follow-up quiz that teaches, not just re-tests.
    Adapts approach based on HOW the learner failed, not just THAT they failed.
    """
    if not api_key:
        api_key = get_api_key()

    if not api_key or not HAS_ANTHROPIC:
        return {"error": "API key required for follow-up quiz generation"}

    client = anthropic.Anthropic(api_key=api_key)

    style_context = ""
    if learner_context:
        style = learner_context.get("learning_style", {})
        teaching_mode = style.get("teaching_mode", "scaffolded")
        style_context = f"""
LEARNER'S TEACHING MODE: {teaching_mode}
- If foundational: Start with the absolute basics. Use analogies. Define terms they may have assumed they understood.
- If scaffolded: Build a sequence where each question teaches a piece, and the final questions combine them.
- If challenging: Reframe the concepts at a higher level - they may have the basics but misunderstand nuances.

LEARNER'S PATTERN: They tend to struggle with {', '.join(style.get('recent_struggles', ['unknown topics']))}
"""

    prompt = f"""You are an adaptive learning tutor. A learner took a quiz and struggled with some concepts.
Your job is NOT just to re-test them - it's to TEACH them through carefully designed questions.

Analyze their mistakes to understand their misconception, then design questions that:
1. First verify the foundational knowledge the concept requires
2. Then build understanding step by step
3. Finally re-test the original concept from a new angle

CONCEPTS THEY STRUGGLED WITH:
{json.dumps(weak_concepts, indent=2)}

THEIR SPECIFIC WRONG ANSWERS (analyze these to find the misconception pattern):
{json.dumps(previous_wrong_answers, indent=2)}
{style_context}
ORIGINAL TRANSCRIPT CONTEXT:
{transcript_text[:40000]}

Generate a TEACHING SEQUENCE of 6-10 questions. Respond with raw JSON (no markdown fencing):
{{
  "quiz_type": "followup",
  "diagnosis": "What misconception pattern you detected from their wrong answers",
  "teaching_strategy": "How this quiz sequence will fix the misunderstanding",
  "focus_areas": ["list of concepts being retested"],
  "quiz": [
    {{
      "question": "...",
      "concept_id": "...",
      "concept_name": "...",
      "topic": "...",
      "difficulty": "easy/medium/hard",
      "bloom_level": "remember/understand/apply/analyze",
      "scaffold_note": "How this question builds toward understanding the missed concept",
      "teaching_moment": "Key insight this question is designed to surface, even if they get it right",
      "options": [
        {{"label": "A", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "B", "text": "...", "correct": true, "why_wrong": null}},
        {{"label": "C", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "D", "text": "...", "correct": false, "why_wrong": "..."}}
      ],
      "explanation": "Rich, clear explanation with an analogy or example",
      "deeper_insight": "Additional context to cement understanding",
      "hint": "A thinking prompt if they're stuck"
    }}
  ]
}}

SEQUENCE DESIGN:
- Questions 1-2: Verify prerequisites (easy, foundational)
- Questions 3-5: Build the concept step by step (medium, scaffolded)
- Questions 6-8: Apply and test from new angles (medium-hard)
- Questions 9-10: Synthesis - combine with other concepts (hard)

Every explanation should TEACH, not just say "B is correct because..."
Use analogies, examples, and connect to things the learner likely already knows."""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=10000,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text.strip()
        return _parse_json_response(response_text)
    except Exception as e:
        return {"error": str(e)}


def generate_review_quiz(due_concepts: list, api_key: str = None) -> dict:
    """Generate a spaced repetition review quiz for concepts due for review."""
    if not api_key:
        api_key = get_api_key()

    if not api_key or not HAS_ANTHROPIC:
        return {"error": "API key required for review quiz generation"}

    client = anthropic.Anthropic(api_key=api_key)

    concept_descriptions = []
    for cid, concept in due_concepts[:15]:
        concept_descriptions.append({
            "id": cid,
            "name": concept.get("name", cid),
            "topic": concept.get("topic", "general"),
            "description": concept.get("description", ""),
            "mastery": concept.get("mastery", 0),
            "times_tested": concept.get("times_tested", 0),
        })

    prompt = f"""Generate a spaced repetition review quiz. These concepts were learned previously and are now
due for review to strengthen long-term retention.

CONCEPTS DUE FOR REVIEW:
{json.dumps(concept_descriptions, indent=2)}

Design questions that test recall from DIFFERENT ANGLES than the learner has seen before.
For concepts with low mastery, make questions slightly easier. For high mastery concepts,
make them harder to push toward deeper understanding.

Respond with raw JSON (no markdown fencing):
{{
  "quiz_type": "review",
  "focus_areas": ["topics covered"],
  "quiz": [
    {{
      "question": "...",
      "concept_id": "...",
      "concept_name": "...",
      "topic": "...",
      "difficulty": "easy/medium/hard",
      "bloom_level": "remember/understand/apply/analyze",
      "options": [
        {{"label": "A", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "B", "text": "...", "correct": true, "why_wrong": null}},
        {{"label": "C", "text": "...", "correct": false, "why_wrong": "..."}},
        {{"label": "D", "text": "...", "correct": false, "why_wrong": "..."}}
      ],
      "explanation": "Clear explanation",
      "deeper_insight": "Connecting this to broader knowledge",
      "hint": "A nudge if stuck"
    }}
  ]
}}"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text.strip()
        return _parse_json_response(response_text)
    except Exception as e:
        return {"error": str(e)}


def _generate_analysis_prompt(transcript_text: str, video_title: str = "") -> dict:
    """
    When no API key is available, return the prompt that the user can
    feed to Cowork or any AI manually.
    """
    return {
        "_source": "prompt_only",
        "message": "No API key configured. Add your Anthropic API key in Settings to enable AI-powered analysis.",
        "prompt": f"Analyze this video transcript...\n\nTITLE: {video_title}\n\nTRANSCRIPT:\n{transcript_text[:5000]}...",
    }
