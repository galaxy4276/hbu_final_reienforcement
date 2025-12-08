
import json
import pandas as pd
import os
import numpy as np
from datetime import datetime

search_paths = [
    "/home/kimjihun/ray_results/halfcheetah_expert_ppo",
    "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2"
]

experiments = []

for base_path in search_paths:
    for root, dirs, files in os.walk(base_path):
        if "params.json" in files:
            exp_data = {}
            folder = root.split('/')[-1]
            try:
                # 1. PARAMS
                with open(os.path.join(root, "params.json"), 'r') as f:
                    params = json.load(f)
                
                parts = folder.split('_')
                if len(parts) > 2:
                    exp_data['ID'] = parts[2]
                    date_str = parts[-2] + " " + parts[-1].replace('-', ':')
                    exp_data['Date'] = date_str[5:16]
                else:
                    exp_data['ID'] = folder[:5]
                    exp_data['Date'] = "Unknown"
                
                # Filter old
                mtime = os.path.getmtime(os.path.join(root, "params.json"))
                dt = datetime.fromtimestamp(mtime)
                if dt.year < 2025 or dt.month < 11: continue
                
                model = params.get('model', {})
                exp_data['Activation'] = model.get('fcnet_activation', 'Unknown')
                
                # 2. METRICS (Find mean and std at peak)
                res_path = os.path.join(root, "result.json")
                
                peak_mean = -np.inf
                std_at_peak = 0.0
                
                if os.path.exists(res_path):
                    with open(res_path, 'r') as rf:
                        for line in rf:
                            try:
                                rec = json.loads(line)
                                # Deep find mean
                                def get_v(d, k_list):
                                    for k in k_list:
                                        if k in d and isinstance(d[k], (int, float)): return d[k]
                                    for v in d.values():
                                        if isinstance(v, dict):
                                            res = get_v(v, k_list)
                                            if res is not None: return res
                                    return None

                                cur_mean = get_v(rec, ['episode_return_mean', 'episode_reward_mean'])
                                
                                if cur_mean is not None:
                                    if cur_mean > peak_mean: 
                                        peak_mean = cur_mean
                                        # Find std
                                        cur_std = get_v(rec, ['episode_return_std', 'episode_reward_std'])
                                        if cur_std: std_at_peak = cur_std
                            except: pass
                            
                if peak_mean != -np.inf:
                    exp_data['Peak Mean'] = f"{peak_mean:.1f}"
                    exp_data['Std'] = f"{std_at_peak:.1f}"
                else:
                    exp_data['Peak Mean'] = "-"
                    exp_data['Std'] = "-"
            
                experiments.append(exp_data)
            except: pass

df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    # Cols
    cols = ['Date', 'ID', 'Activation', 'Peak Mean', 'Std']
    print(df[cols].to_markdown(index=False))
else:
    print("None")
