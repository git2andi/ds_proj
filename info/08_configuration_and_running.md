# Configuration and running

## LLM credentials

The runtime loads credentials from a `.env` file in the repository root through `python-dotenv`. Add only the key required by the provider selected in `config.yaml`:

```dotenv
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

Provider `uni` uses the configured HTTP endpoint and does not require one of these API keys. Never commit `.env`; it is listed in `.gitignore`. An `.env.example` file may be committed because it contains placeholders only.

## Configuration

`config.yaml` controls the implemented runtime. Its main sections are:

- `llm`: active dialogue provider, per-provider model IDs, endpoint, timeout, and Gemini request delay;
- `environment`: automatic or manual scenario;
- `simulation`: participant count, supported bounds, seed, and setup-attempt limit;
- `participants`: automatic or manual profiles;
- `scenario`: option labels, public-attribute bounds, context bounds, and alias length;
- `personas`: trait ranges, hard-blocker probability, and preference-shape weights;
- `conversation`: turn budgets, thread cap, stagnation threshold, narrowing window, prompt context, and consecutive-turn bound;
- `simulator`: engagement and movement probabilities;
- `language`: verbosity limits, action-specific caps, and directness instructions;
- `moderator`, `consensus`, `limits`, and `output`.

Scenario and persona generation each use `simulation.setup_generation_attempts`. The separate alias-and-name metadata call does not consume another scenario attempt and cannot invalidate an already accepted board. Only openings use LLM repair. Final vote wording is deterministic.

## Install and run

```powershell
py -m pip install -r requirements.txt
py .\main.py
```

Deterministic tests:

```powershell
py -m pytest -q
```

Small scenario batch and deterministic summary:

```powershell
py .\eval\run_scenarios.py --limit 10 --seed 500 --clean
py .\eval\summarize_runs.py --logs .\eval\logs_scenarios
```

The scenario runner uses two isolated worker processes by default. Use `--workers 1` for sequential execution. It refuses to write into a nonempty output directory unless `--clean` is supplied.

Post-hoc transcript judging:

```powershell
py .\eval\judge_transcripts.py --logs .\eval\logs_scenarios --judges 3 --provider uni
```

The judge uses two workers by default and resumes from existing complete panels.
