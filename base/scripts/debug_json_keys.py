
import json
import os

path = "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_d9bbc_00000_0_2025-12-03_16-22-39/result.json"

with open(path, 'r') as f:
    # Read last line
    lines = f.readlines()
    if not lines:
        print("Empty file")
        exit()
    last_line = lines[-1]
    data = json.loads(last_line)
    
    # Recursive search for keys
    def search_keys(d, prefix=""):
        for k, v in d.items():
            curr_key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                search_keys(v, curr_key)
            else:
                if "mean" in k.lower() or "reward" in k.lower() or "return" in k.lower():
                    print(f"{curr_key}: {v}")

    search_keys(data)
