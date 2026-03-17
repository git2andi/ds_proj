from src.ds_proj.profiles import get_default_profiles

profiles = get_default_profiles()

total_share = sum(p["target_turn_share"] for p in profiles)

print("Profiles:")
for p in profiles:
    print(p)

print()
print("Total target turn share:", total_share)