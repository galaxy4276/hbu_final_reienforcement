
import json
import os

path = "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_0b810_00000_0_2025-12-07_02-21-47/result.json"

with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        # Recursive print keys with "std"
        def print_std(d, p=""):
            for k, v in d.items():
                if "std" in k.lower():
                    print(f"{p}{k}: {v}")
                if isinstance(v, dict):
                    print_std(v, p+k+"/")
        print_std(data)
        break # just check first line or last? Check last.

print("--- Last Line ---")
with open(path, 'r') as f:
    lines = f.readlines()
    if lines:
        data = json.loads(lines[-1])
        def print_std(d, p=""):
            for k, v in d.items():
                if "std" in k.lower(): # Only print if value is scalar
                    if isinstance(v, (int, float)):
                         print(f"{p}{k}: {v}")
                if isinstance(v, dict):
                    print_std(v, p+k+"/")
        print_std(data)
