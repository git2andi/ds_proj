from pathlib import Path

from src.ds_proj.normalize import normalize_real_dialogue, normalize_simulated_dialogue

real_path = Path(r"C:\Users\andi\Desktop\ds_proj\data\delidata\raw\all\00639b9b-3a45-4c59-90f6-a4738de92fd4.tsv")
sim_path = Path(r"C:\Users\andi\Desktop\ds_proj\outputs\simulated_dialogue_001.json")

real_df = normalize_real_dialogue(real_path)
sim_df = normalize_simulated_dialogue(sim_path)

print("REAL:")
print(real_df.head(5).to_string(index=False))
print()
print("SIMULATED:")
print(sim_df.head(5).to_string(index=False))