from pathlib import Path
from src.ds_proj.delidata_loader import load_delidata_dialogue

file_path = Path(r"C:\Users\andi\Desktop\ds_proj\data\delidata\raw\all\00639b9b-3a45-4c59-90f6-a4738de92fd4.tsv")

dialogue = load_delidata_dialogue(file_path)

print(dialogue.head(10).to_string(index=False))
print()
print("Turns:", len(dialogue))
print("Speakers:", dialogue["speaker"].nunique())
print("Speaker names:", sorted(dialogue["speaker"].unique().tolist()))