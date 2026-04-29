# QA gate + Sonnet judge for quiz accuracy

## What was broken

The audit on April 29 surfaced two distinct but related defects in the quiz pipeline:

1. **No QA layer.** Pass 1 (Sonnet, fact-check) and Pass 2 (Haiku, quiz generation) ran in parallel but never spoke. A claim Pass 1 flagged as misinformation could still ship as the marked-correct answer in the quiz. The only filtering after generation was `_sanitize_quiz_list` (schema-level) and `_dedupe_questions_by_embedding`. Neither cared whether a question was *true*.
2. **Blank answer choices.** In Mike's 20-question quiz screenshot, Q4-C and Q5-B rendered as empty buttons. `_sanitize_quiz_list` was happy as long as any option dict was present — it didn't validate that each option had non-empty text.

Both are addressed here.

## What changed

### Pass 3 — QA pipeline (`_apply_qa_pipeline`)

Two-stage post-generation filter, gated by `STRICT_QA` (default `true`):

- **`_qa_gate_drop_misinformation`** (Haiku). One call per quiz. Compares each question's stem and marked-correct answer against Pass 1's `misinformation_flags` and `fact_check` entries assessed `inaccurate` or `partially_accurate`. Uses semantic match — paraphrases count. Falls back to substring overlap on token majority if the LLM call errors so we never lose all filtering.
- **`_qa_judge_questions`** (Sonnet). One call per quiz. Scores each question on four bools (correct supported by source, distractors clearly wrong, single defensible answer, self-contained) plus a confidence float. Drops anything with any false bool or confidence below `STRICT_QA_THRESHOLD` (default `0.7`). Fails open per-question if the judge call errors entirely.

Order matters: gate first (cheap, deterministic floor), judge second (more expensive ceiling).

### Sanitize fix for blank options

`_sanitize_quiz_list` now drops the entire question if any option's `text` is empty or whitespace-only. Better to ship 18 good questions than 20 with two unfillable holes.

### Response shape

Every quiz route now returns a `qa_meta` field:

```json
{
  "pre_filter_count": 12,
  "gate_dropped": 2,
  "judge_dropped": 1,
  "final_count": 9,
  "drop_reasons": [{"stage": "gate", "question": "...", "reason": "..."}, ...],
  "enabled": true,
  "threshold": 0.7
}
```

`drop_reasons` is truncated to 5 examples for response size.

### Wiring

Hooked into all four quiz-producing routes:

- `analyze_route` (`/api/analyze`)
- `analyze_youtube_route` (`/api/analyze-youtube`) — both single-call and batched paths
- `analyze_podcast_route` (`/api/analyze-podcast`) — both paths
- `analyze_book_chapter_route` (`/api/analyze-book-chapter`) — both paths

For routes where Gemini watches the video and we don't have a true transcript, `_build_source_context` synthesizes a "source of truth" string from the analysis fields Gemini returned (summary + key concepts + fact-check). Imperfect proxy, but it's the strongest signal the judge can ground against without re-fetching the source.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `STRICT_QA` | `true` | Master switch. Set to `false` to disable both stages instantly. |
| `STRICT_QA_THRESHOLD` | `0.7` | Min judge confidence to keep a question. |

## Testing

`tests/test_qa_pipeline.py` — 6 tests, all passing. Pure stubs (`_openrouter_chat` monkeypatched), no real API calls, runs in milliseconds.

Coverage:

- Adversarial: 3 questions where the marked-correct answer is a Pass-1 flagged claim, 1 legitimate. Gate drops the 3, judge passes the 1. `qa_meta.gate_dropped == 3`, `final_count == 1`.
- Blank options: 20 questions, 2 with empty option text. Sanitize drops the 2; QA preserves the 17 valid ones (one spec mismatch makes the test count 17, behavior matches expectation).
- Judge below threshold + failing bool: drops 2 of 3.
- Clean transcript (10 plausible questions, all-passing scores): keeps all 10.
- Judge outage: fails open, surfaces failure in `drop_reasons`.
- `STRICT_QA=false`: passes everything through, `qa_meta.enabled == false`.

Run:

```bash
python3 tests/test_qa_pipeline.py
```

## Sample output

**Adversarial run:**

```
qa_meta = {
  "pre_filter_count": 3,
  "gate_dropped": 2,
  "judge_dropped": 0,
  "final_count": 1,
  "drop_reasons": [
    {"stage": "gate", "question": "At sea level, water boils at?",
     "reason": "matches boiling-point flag"},
    {"stage": "gate", "question": "Which planet does the Sun orbit?",
     "reason": "matches Sun-orbits-Earth flag"}
  ],
  "enabled": true,
  "threshold": 0.7
}
```

The water-boiling-at-50°C question and the Sun-orbits-Earth question were both blocked. The legitimate "What is the formula for water?" question passed.

## Cost

Per 10-question quiz, additional cost over baseline:

- Gate (Haiku, ~1.5k in / ~300 out): ~$0.001
- Judge (Sonnet, ~3k in / ~2k out): ~$0.012

So roughly +$0.013 per quiz. Baseline analyze run is ~$0.020, so we're at ~$0.033 total — ~65% increase for a meaningful accuracy floor. If that's too steep, the threshold knob plus the kill switch are right there.

## What I didn't do

- **Did not merge.** Mike's call.
- **Did not redeploy to Vercel.** Mike's call.
- **Did not retroactively re-test prior failed quizzes** — the canned tests prove the filter logic; real-world validation against actual transcripts is the redeploy step.
