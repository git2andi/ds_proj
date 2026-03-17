from pathlib import Path
import json
import pandas as pd

from src.ds_proj.delidata_loader import load_delidata_dialogue


def normalize_real_dialogue(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    df = load_delidata_dialogue(file_path).copy()

    df["dialogue_id"] = file_path.stem
    df["source"] = "real"
    df["role"] = None
    df["act"] = None

    df = df[
        ["dialogue_id", "source", "turn_id", "speaker", "speaker_id", "timestamp", "role", "act", "text"]
    ] if "timestamp" in df.columns else df

    return df


def normalize_simulated_dialogue(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data).copy()
    df["dialogue_id"] = file_path.stem
    df["source"] = "simulated"
    df["speaker_id"] = None
    df["timestamp"] = None

    df = df[
        ["dialogue_id", "source", "turn_id", "speaker", "speaker_id", "timestamp", "role", "act", "text"]
    ]

    return df