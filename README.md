# Option-Grounded Multi-User Simulator

This project generates bounded discussions between two to seven autonomous user simulators. A structured policy controls participant decisions, floor allocation, visible stance movement, narrowing, and voting; an LLM realizes selected discussion actions as text.

## Setup

Install the dependencies:

```powershell
py -m pip install -r requirements.txt
```

Create a `.env` file in the repository root and add the key for the provider selected in `config.yaml`:

```dotenv
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

Only one key is required when only one hosted provider is used. The `uni` provider uses the endpoint configured under `llm.endpoints.uni` and does not require one of these keys. `.env` is ignored by Git and must not be committed.

Select the dialogue provider, model, endpoint, participant count, and other runtime values in `config.yaml`.

## Run

```powershell
py .\main.py
```

Each run writes a readable transcript and a structured `run.json` to the configured log directory.

## Tests and evaluation

```powershell
py -m pytest -q
py .\eval\run_scenarios.py --limit 10 --seed 500 --clean
py .\eval\summarize_runs.py --logs .\eval\logs_scenarios
```

See [`eval/README.md`](eval/README.md) for the full evaluation commands and [`info/`](info/) for implementation details.
