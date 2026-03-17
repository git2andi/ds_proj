from pathlib import Path
import pandas as pd

from src.ds_proj.delidata_loader import load_delidata_dialogue


data_dir = Path(r"C:\Users\andi\Desktop\ds_proj\data\delidata\raw\all")
files = sorted(data_dir.glob("*.tsv"))

rows = []

# start small: first 50 files only
for file_path in files[:50]:
    try:
        dialogue = load_delidata_dialogue(file_path)
    except Exception as e:
        print(f"Skipping {file_path.name}: {e}")
        continue

    if len(dialogue) == 0:
        continue

    speaker_counts = dialogue["speaker"].value_counts()
    num_speakers = dialogue["speaker"].nunique()
    num_turns = len(dialogue)

    # simple participation imbalance:
    # fraction of turns from the most active speaker
    top_speaker_fraction = speaker_counts.iloc[0] / num_turns

    rows.append({
        "file": file_path.name,
        "num_turns": num_turns,
        "num_speakers": num_speakers,
        "top_speaker_fraction": top_speaker_fraction,
    })

stats = pd.DataFrame(rows)

print("Number of dialogues analyzed:", len(stats))
print()

print("Turns per dialogue:")
print(stats["num_turns"].describe())
print()

print("Speakers per dialogue:")
print(stats["num_speakers"].value_counts().sort_index())
print()

print("Top-speaker fraction:")
print(stats["top_speaker_fraction"].describe())
print()

print("Example dialogues:")
print(stats.head(10).to_string(index=False))