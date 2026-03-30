# LearnEngine Deep Audit

**Date:** March 29, 2026
**Scope:** Full codebase review — architecture, security, data integrity, error handling, frontend, API design

---

## Executive Summary

LearnEngine is a Flask-based adaptive learning system that converts YouTube videos into AI-analyzed quizzes with spaced repetition tracking. The concept is strong and the core flow works. But there are several issues ranging from a critical security hole to architectural bottlenecks that will bite hard as usage scales even slightly. This audit covers 6 files of application code, 2 HTML quiz artifacts, and 2 data files.

**Severity breakdown:** 3 critical, 5 high, 8 medium, 6 low.

---

## Critical Issues

### 1. API Key Stored in Plaintext JSON (analyzer.py:36-46)

The Anthropic API key gets written to `data/config.json` as a plain string. Anyone with filesystem access (or any process that can read the data directory) can extract it. There's no encryption, no file permissions restriction, no `.gitignore` entry protecting it.

The `/api/config` GET endpoint (app.py:224-237) also leaks a partial key preview over HTTP. If this ever runs on a non-localhost network, that's a direct credential exposure vector.

**Fix:** Use environment variables exclusively, or at minimum encrypt at rest with a machine-specific key. Drop the preview endpoint entirely — knowing the first 8 and last 4 characters of an API key is more information than any client needs.

### 2. Debug Mode in Production (app.py:967)

```python
app.run(host="0.0.0.0", port=5050, debug=True)
```

This binds to all interfaces with Flask's debug mode enabled. Debug mode exposes the Werkzeug interactive debugger, which allows arbitrary Python code execution from the browser. Combined with `0.0.0.0` binding, anyone on the local network can get a shell on the machine.

**Fix:** `debug=False` by default, configurable via environment variable. Bind to `127.0.0.1` unless explicitly overridden.

### 3. Stored XSS via Quiz Content (app.py, entire frontend)

The frontend renders all content — quiz questions, explanations, video titles, fact-check claims, misinformation flags — directly into innerHTML via template literals. None of it is sanitized.

A malicious transcript or a Claude API response containing `<script>` tags or event handlers will execute in the user's browser. Since the app also handles API keys in the same session, this is a direct path from "paste a YouTube URL" to "steal the user's API key."

Specific locations: `renderQuiz()` (line ~650), `renderResults()` (line ~710), `renderHistory()` (line ~769), and essentially every render function that interpolates `${}` into HTML.

**Fix:** Create an `escapeHtml()` utility and apply it to every dynamic string before inserting into the DOM. Or switch to `textContent` assignment and build DOM nodes programmatically.

---

## High Severity Issues

### 4. No Input Validation on API Endpoints (app.py:25-93)

The `/api/process` endpoint accepts arbitrary URLs and transcript text with zero validation. There's no URL format check beyond what `extract_video_id` does (which silently fails), no transcript length limit enforced at the API layer, and no rate limiting. A user can POST megabytes of transcript text and it gets forwarded directly to Claude with only a character truncation in `analyzer.py`.

The `/api/submit_quiz` endpoint doesn't validate that answer keys match actual question indices. The `/api/config` POST accepts any string as an API key with no format validation.

### 5. Race Conditions on JSON File Storage (quiz_engine.py)

Every read-modify-write cycle on JSON files (videos.json, quizzes.json, learner_profile.json) is non-atomic. `load_json` reads, the caller modifies, `save_json` writes. If two requests hit `/api/process` simultaneously, one write clobbers the other. This isn't theoretical — the `init()` function on the frontend fires 4 parallel API requests on page load.

**Fix:** Use file locking (`fcntl.flock` or a library like `filelock`), or migrate to SQLite which handles concurrent access natively.

### 6. Unbounded Data Growth (quiz_engine.py)

`videos.json` and `quizzes.json` grow without limit. Every processed video appends its full transcript, full analysis, and full quiz to these files. A single CrashCourse video transcript is thousands of words. After 50 videos, these JSON files will be substantial; after a few hundred, load times will degrade noticeably since the entire file is parsed on every read.

The learner profile caps history at 100 entries, but `concepts` and `topics` dictionaries grow unbounded.

### 7. Entire Frontend in a Python String (app.py:247-962)

960 lines of HTML/CSS/JavaScript embedded as a raw string inside a Python file. This makes the frontend essentially impossible to lint, format, syntax-check, or work on with standard tooling. Template literals inside a Python string means you can't use f-strings or standard Flask templating without collision.

More practically: every render cycle blows away the entire DOM via `innerHTML`, which kills scroll position, focus state, and any in-progress user interaction. The `selectAnswer` function triggers a full re-render, meaning on a 12-question quiz the user experiences 12 full DOM rebuilds just answering questions.

---

## Medium Severity Issues

### 8. SM-2 Implementation Oversimplification (quiz_engine.py:170-179)

Quality scores are binary: correct = 5, wrong = 1. The SM-2 algorithm was designed for a 0-5 scale where partial recall (3-4) produces meaningfully different scheduling than perfect recall (5). Collapsing to two values makes the easiness factor converge toward extremes and produces suboptimal review intervals. A user who hesitated but got the answer right should not be scheduled identically to one who answered instantly.

### 9. No CORS Configuration (app.py)

No CORS headers are set. This means the API will reject cross-origin requests, which is actually fine for localhost-only use. But if the app is ever served behind a reverse proxy or on a different port from a separate frontend, it breaks silently. More importantly, there's no CSRF protection either — any page the user visits while the app is running can POST to the API.

### 10. Error Swallowing in Transcript Fetcher (transcript.py:72-80)

The catch-all `except Exception` returns a dict with an error string instead of raising. This is fine for the happy path, but the error dict has `full_text: None`, and downstream code in `app.py:55-58` calls `analyze_transcript(transcript_data["full_text"])`, which would pass `None` to Claude if the error check on line 48-53 somehow missed it.

### 11. JSON Parsing Fragility (analyzer.py:144-159)

The markdown fence stripping logic assumes fences are on their own lines and uses a simple string split. If Claude returns ` ```json\n{...}\n``` ` with trailing whitespace or a language tag variation, the strip fails and `json.loads` throws. The error handler on line 155 references `response_text` via `'response_text' in dir()` which is a Python anti-pattern (should be a try/except or local variable check).

### 12. No Transcript Caching

Processing the same YouTube URL twice fetches the transcript again and calls the Claude API again. There's no deduplication check at the API layer — `save_video_data` in quiz_engine.py does update-if-exists for video data, but the expensive API call has already been made by that point.

### 13. Hardcoded Model Version (analyzer.py:137, 227)

`claude-sonnet-4-20250514` is hardcoded in two places. When Anthropic releases newer model versions, both need manual updates. Should be a config constant or pulled from environment.

### 14. No Health Check or Startup Validation

The app starts and serves requests without verifying that the data directory is writable, that dependencies are the right version, or that the Anthropic SDK is actually importable (it gracefully degrades, but the user gets no feedback about why analysis isn't working beyond a vague "prompt_only" response).

### 15. Frontend State Not Persisted

Quiz progress (which answers you've selected) lives only in JavaScript memory. Refreshing the page during a 12-question quiz loses all your selections. The state object should checkpoint to sessionStorage or the backend.

---

## Low Severity Issues

### 16. `requirements.txt` Version Pinning Too Loose

`flask>=3.0` and `anthropic>=0.40.0` will happily install future major versions that may break the app. Pin to compatible ranges (`~=3.0` or `>=3.0,<4.0`).

### 17. `__pycache__` in the Project Directory

Should be `.gitignore`d. It's present in the folder, suggesting either no `.gitignore` exists or it's misconfigured.

### 18. `start.sh` Suppresses pip Errors

The `2>/dev/null` redirect on the first pip attempt means if installation fails for a real reason (network, permissions), the user sees no output — the script just falls through to the un-silenced attempt.

### 19. Orphaned Quiz Data Files

`data/video_lkXFb1sMa38.json` exists as a standalone file outside the `videos.json` collection. There's no code that reads individual video JSON files — this appears to be a manual artifact. Creates confusion about what's authoritative.

### 20. No Favicon or Meta Tags

Minor, but the app will generate 404s for favicon requests on every page load, cluttering server logs.

### 21. `is_generated` Hardcoded to False (transcript.py:67)

The transcript fetcher always sets `is_generated: False` regardless of whether YouTube's auto-generated captions were used. The youtube-transcript-api can distinguish these; the field should reflect reality since auto-generated transcripts have meaningfully lower accuracy.

---

## Architectural Observations

The two-mode design (standalone API calls vs. "Cowork prompt generation") is clever but creates a confusing UX. When there's no API key, the user gets a `_source: "prompt_only"` response that the frontend doesn't know how to render — it just shows an error. The fallback path needs its own UI flow.

The spaced repetition system is well-structured but disconnected from the rest of the app. There's no UI for reviewing due concepts — the `/api/due_reviews` endpoint exists but nothing in the frontend calls it. The "Due for Review" stat card on the dashboard is a dead number.

The quiz HTML files (`quiz_1960s_crashcourse.html` and `quiz_1960s_v2.html`) appear to be standalone artifacts from an earlier iteration, not generated by the current app. They're 22KB and 71KB respectively and share no code with the Flask frontend. If they're not in active use, they're dead weight creating false expectations about the app's output format.

---

## Recommended Priority Order

1. Fix debug mode and network binding (5 minutes, eliminates RCE risk)
2. Add HTML escaping to all dynamic content (30 minutes, eliminates XSS)
3. Move API key to environment variable only (15 minutes, eliminates credential leak)
4. Add SQLite or file locking for data persistence (2-3 hours, eliminates data loss)
5. Extract frontend to separate files with a proper build step (half day)
6. Add input validation and rate limiting on all endpoints (1-2 hours)
7. Implement transcript caching / dedup (1 hour)
8. Wire up the spaced repetition review UI (2-3 hours)
