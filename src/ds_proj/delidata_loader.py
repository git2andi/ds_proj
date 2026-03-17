from pathlib import Path
import ast
import pandas as pd

def extract_message(content: str) -> str:
    if pd.isna(content):
        return ""
    try:
        parsed = ast.literal_eval(content)
        if isinstance(parsed, dict):
            return str(parsed.get("message", "")).strip()
    except Exception:
        pass
    return str(content).strip()


def load_delidata_dialogue(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    df = pd.read_csv(file_path, sep="\t", dtype=str)

    required_cols = ["MESSAGE_TYPE", "CONTENT", "TIMESTAMP", "USER_NAME", "USER_ID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    chat_df = df[df["MESSAGE_TYPE"] == "CHAT_MESSAGE"].copy()
    chat_df["text"] = chat_df["CONTENT"].apply(extract_message)
    chat_df = chat_df[chat_df["text"] != ""].copy()

    out = chat_df[["TIMESTAMP", "USER_NAME", "USER_ID", "text"]].copy()
    out = out.rename(columns={
        "TIMESTAMP": "timestamp",
        "USER_NAME": "speaker",
        "USER_ID": "speaker_id",
    })

    out = out.sort_values("timestamp").reset_index(drop=True)
    out.insert(0, "turn_id", range(len(out)))

    return out