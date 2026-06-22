# Evaluation Workflow

How to evaluate the group-discussion simulator after a code change.

## Quick reference

```powershell
# 1. Run unit tests (no network, instant)
py -m pytest tests/ -v

# 2. Run a live eval spread (requires VPN for uni provider)
& .\dspro\Scripts\python.exe evals\run_eval.py

# 3. Run a single topic manually and read the transcript
"Plan a weekend team offsite" | & .\dspro\Scripts\python.exe .\main.py
# then read logs/<newest>/transcript.md
```

## Layers

### Layer 1: Unit tests (`tests/`)

Fast, offline, deterministic. Covers the code that changes most often:

- **`test_validation.py`** — echo guard, robotic phrasing, opener variety, question
  chain, speaker prefix, empty turn, invented numbers, collective voice.
- **`test_parsing.py`** — trailer extraction (bracketed, bare, missing), commitment
  gating (decision acts vs discussion), hedged-accept detection, option resolution.

Run after every code change. If these fail, don't bother with a live run.

### Layer 2: Post-run automated checks (`evals/run_eval.py`)

Reads `run.json` files from completed runs and checks for regressions:

- **Opener variety**: % of participant turns starting with I/I'm/we/our (threshold: <=50%).
- **Repair rate**: fraction of turns that needed repair (informational, not a gate).
- **Duplicate moderator lines**: exact-match count (threshold: 0).
- **Hard-blocker integrity**: a hard-blocker persona should never vote/accept a non-preferred option.
- **Mid-discussion binding accepts**: turns in discussion phase recorded as binding accept (threshold: 0).
- **Robotic template count**: turns with `ROBOTIC_TEMPLATE` or `POSSESSIVE_SUBJECT` still in final issues.
- **Outcome sanity**: consensus requires support fraction == 1.0; fallback requires >= 0.66.

The script can run against existing logs or drive new runs from `evals/scenarios.yaml`.

### Layer 3: Manual transcript review

Read `logs/<run_id>/transcript.md` for:

- **Naturalness**: Do people react to each other or just state positions?
- **Adjacency pairs**: Is a question answered by the next relevant speaker?
- **Stance grounding**: When someone changes their mind, is there a visible trigger?
- **Moderator accuracy**: Does the moderator name the right front-runners?
- **Closure quality**: Does the outcome match what the transcript showed?

Use `run.json` to cross-check: per-turn `act_type`, `validation_issues`, `repaired`.

## Scenarios

`evals/scenarios.yaml` contains a spread of topics and group sizes used across
evaluation passes. When adding a new topic, pick a domain not already covered and
include at least one small (2-3) and one large (5-7) group size.

## Metrics to track across runs

All logged in `logs/metrics.csv`:

| Metric | What it tells you |
|---|---|
| `outcome_status` | consensus / fallback / unresolved |
| `final_support_fraction` | 1.0 = full consensus, <1 = some holdouts |
| `repaired_turns` / `repair_rate` | how often validation had to fix a turn |
| `flagged_turns` | turns with any remaining issue (usually warn-level) |
| `question_density` | fraction of turns containing a question |
| `avg_words_per_turn` | turn length — too high = monologue, too low = empty |
| `min_discussion_turns` | derived pacing target for this run |
| `option_coverage` | per-option mention/reason/objection/acceptance counts |

## Regression checklist

After a code change:

1. `py -m pytest tests/ -v` — all green.
2. Run at least one n=3 and one n=6 topic via `run_eval.py` or manual pipe.
3. Check `run_eval.py` output for any FAIL lines.
4. Skim the newest transcript for naturalness.
5. If the change touched `validation.py` or `parsing.py`, verify the specific check
   with its unit test and at least one live run that exercises it.
