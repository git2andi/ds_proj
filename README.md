# Dialogue Simulator

Generates multi-participant group discussions driven by LLM-backed personas.
A moderator facilitates the discussion toward a consensus decision.

---

## Project Structure

```
dialogue_sim/
│
├── config.yaml          ← All tuneable parameters. Start here.
├── scenarios.txt        ← One topic per line for batch runs.
│
├── main.py              ← Entry point (interactive or batch)
├── config_loader.py     ← Loads config.yaml → exposes `cfg`
├── prompts.py           ← Every LLM prompt in one place
├── llm_client.py        ← Provider abstraction (Gemini / Groq / Ollama)
├── persona.py           ← Persona dataclass + PersonaBuilder
├── simulator.py         ← Wraps a Persona; generates one turn
├── turn_manager.py      ← Speaker selection + repetition detection
├── consensus.py         ← Three-tier consensus detection
├── orchestrator.py      ← Coordinates one dialogue run
└── logger.py            ← .txt transcript + .csv data file output
```

---

## How It Works

```
main.py
  └─ run_dialogue(topic)
       ├─ Orchestrator          generates 4 options via LLM
       ├─ PersonaBuilder        assigns roles + builds N personas (N+1 LLM calls)
       ├─ Simulator × N        wraps each persona
       └─ Orchestrator.run_simulation()
            ├─ TurnManager     selects who speaks each round
            ├─ Simulator       generates each turn (1 LLM call per turn)
            ├─ ConsensusDetector  checks for agreement (soft → regex → LLM)
            └─ DialogueLogger  writes .txt + .csv to logs/
```

---

## Usage

### Interactive (single dialogue)
```bash
python main.py
# You will be prompted: "Enter the dialogue topic:"
```

### Batch (multiple dialogues from a file)
```bash
python main.py scenarios.txt
```

`scenarios.txt` format — one topic per line, `#` for comments:
```
Plan a birthday party for a colleague
Book a group flight to Stockholm
# This line is ignored
Should we go out tonight?
```

---

## Configuration (`config.yaml`)

| Section | Key | Effect |
|---|---|---|
| `llm` | `provider` | `"uni"` / `"groq"` / `"gemini"` |
| `simulation` | `num_participants` | Fixed group size |
| `simulation` | `num_participants_random` | If `true`, picks randomly from min/max |
| `simulation` | `moderator_style` | `"active"` / `"minimal"` / `"passive"` |
| `turns` | `hard_ceiling` | Max turns before force-close |
| `turns` | `min_before_narrowing` | Participant turns before narrowing allowed |
| `consensus` | `llm_check_every_n_turns` | How often the LLM checks for consensus |
| `personas` | `generate_backstory` | Whether to LLM-generate character backstory |
| `output` | `log_dir` | Where transcripts are saved |
| `output` | `save_csv` / `save_txt` | Toggle output formats |

---

## Prompts (`prompts.py`)

All text sent to the LLM lives in `prompts.py`. To change how the moderator
speaks, how personas are built, or how turns are phrased — edit that file only.

| Function | Purpose |
|---|---|
| `option_generation` | Generate 4 options + opening question |
| `role_assignment` | Assign topic-aligned roles to all participants |
| `persona_concept` | Generate backstory, goal, personality for one participant |
| `sim_turn` | The per-turn prompt each simulator uses |
| `consensus_check` | LLM fallback for agreement detection |
| `moderator_intervention` | Generic moderator line (stalls, outliers, silent participants) |
| `moderator_clarification` | Moderator corrects speculative loops about option details |

---

## Scaling Up

To run 1000 dialogues:
1. Create a `scenarios.txt` with 1000 lines (one topic each).
2. Run: `python main.py scenarios.txt`
3. All logs land in `logs/` as `<timestamp>.txt` and `<timestamp>.csv`.

To vary group size per dialogue, set in `config.yaml`:
```yaml
simulation:
  num_participants_random: true
  num_participants_min: 2
  num_participants_max: 5
```

---

## Output

Each dialogue produces two files in `logs/`:

- `<id>.txt` — human-readable transcript
- `<id>.csv` — one row per turn, with speaker, phase, persona traits, and text

The CSV is designed for downstream analysis (e.g. pandas, R).

---

## LLM calls per dialogue

| Step | Calls |
|---|---|
| Option generation | 1 |
| Role assignment | 1 |
| Persona concepts | N (one per participant) |
| Turn generation | ~15–40 (depends on dialogue length) |
| Consensus checks | ~3–8 |
| Moderator lines | ~2–6 |
| **Total** | **~25–60 per dialogue** |
