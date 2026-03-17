from pathlib import Path
import pandas as pd
import ast

file_path = Path(r"C:\Users\andi\Desktop\ds_proj\data\delidata\raw\all\00639b9b-3a45-4c59-90f6-a4738de92fd4.tsv")

# Load TSV
df = pd.read_csv(file_path, sep="\t", dtype=str)

print("Columns:")
print(df.columns.tolist())
print()

print("MESSAGE_TYPE counts:")
print(df["MESSAGE_TYPE"].value_counts(dropna=False))
print()

# Keep only actual chat messages
chat_df = df[df["MESSAGE_TYPE"] == "CHAT_MESSAGE"].copy()

def extract_message(content: str) -> str:
    if pd.isna(content):
        return ""
    try:
        parsed = ast.literal_eval(content)
        if isinstance(parsed, dict):
            return str(parsed.get("message", ""))
    except Exception:
        pass
    return str(content)

chat_df["text"] = chat_df["CONTENT"].apply(extract_message)

# Normalize a compact view
out = chat_df[["TIMESTAMP", "USER_NAME", "USER_ID", "text"]].copy()
out = out.rename(columns={"USER_NAME": "speaker", "USER_ID": "speaker_id"})
out = out.reset_index(drop=True)
out.insert(0, "turn_id", range(len(out)))

print("First chat turns:")
print(out.head(20).to_string(index=False))
print()

print(f"Total chat turns: {len(out)}")
print(f"Unique speakers: {out['speaker'].nunique()}")
print("Speakers:", sorted(out["speaker"].dropna().unique().tolist()))