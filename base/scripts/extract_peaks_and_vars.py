
import json
import pandas as pd
import os
import numpy as np
from datetime import datetime

search_paths = [
    "/home/kimjihun/ray_results/halfcheetah_expert_ppo",
    "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2"
]

all_params = []
experiments = []

# Helper to flatten dict
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

for base_path in search_paths:
    for root, dirs, files in os.walk(base_path):
        if "params.json" in files:
            try:
                # 1. PARAMS
                with open(os.path.join(root, "params.json"), 'r') as f:
                    p = json.load(f)
                
                # Manual exclusions/adjustments for readability
                # e.g. remove nulls
                flat_p = flatten(p)
                # Filter out uninteresting keys (callbacks, logging, etc)
                filtered_p = {k: v for k,v in flat_p.items() if not any(x in k for x in ['callback', 'log', 'input', 'output', 'framework', 'resource', 'env_config'])}
                
                # Special formatting for list/dict values to make them hashable/comparable
                for k, v in filtered_p.items():
                    if isinstance(v, list): filtered_p[k] = str(v)
                
                # Checkpoint Info
                folder = root.split('/')[-1]
                parts = folder.split('_')
                checkpoint_id = parts[2] if len(parts)>2 else folder[:5]
                filtered_p['Checkpoint'] = checkpoint_id
                
                # Date
                mtime = os.path.getmtime(os.path.join(root, "params.json"))
                dt = datetime.fromtimestamp(mtime)
                if dt.year < 2025 or dt.month < 12: continue
                if dt.day < 2: continue
                filtered_p['Date'] = dt.strftime('%m-%d %H:%M')
                
                all_params.append(filtered_p)
                
                # 2. METRICS (Finding Peaks)
                res_path = os.path.join(root, "result.json")
                metrics = {'Mean': -np.inf, 'Max': -np.inf, 'Min': -np.inf}
                
                if os.path.exists(res_path):
                    with open(res_path, 'r') as rf:
                        for line in rf:
                            try:
                                record = json.loads(line)
                                # Find reward keys (recursively or standard)
                                def get_val(d, keys):
                                    for k in keys:
                                        if k in d: return d[k]
                                        # deep search
                                        for subk, subv in d.items():
                                            if isinstance(subv, dict):
                                                res = get_val(subv, [k])
                                                if res is not None: return res
                                    return None
                                
                                mean_val = get_val(record, ['episode_return_mean', 'episode_reward_mean'])
                                max_val = get_val(record, ['episode_return_max', 'episode_reward_max'])
                                min_val = get_val(record, ['episode_return_min', 'episode_reward_min'])
                                
                                if mean_val is not None and mean_val > metrics['Mean']: metrics['Mean'] = mean_val
                                if max_val is not None and max_val > metrics['Max']: metrics['Max'] = max_val
                                if min_val is not None and min_val > metrics['Min']: metrics['Min'] = min_val
                            except: pass
                
                # Add metrics to dict
                filtered_p['Peak_Mean'] = metrics['Mean'] if metrics['Mean'] != -np.inf else float('nan')
                filtered_p['Peak_Max'] = metrics['Max'] if metrics['Max'] != -np.inf else float('nan')
                filtered_p['Peak_Min'] = metrics['Min'] if metrics['Min'] != -np.inf else float('nan')
                
                experiments.append(filtered_p)
                
            except Exception as e:
                pass

# Determine changing params
df = pd.DataFrame(experiments)
if df.empty:
    print("No data")
    exit()

# Identify columns with >1 unique val
changing_cols = []
ignored_cols = ['Checkpoint', 'Date', 'Peak_Mean', 'Peak_Max', 'Peak_Min']
for col in df.columns:
    if col in ignored_cols: continue
    if df[col].astype(str).nunique() > 1:
        changing_cols.append(col)

# Sort vars
# Prioritize known important ones
priority = ['model_fcnet_hiddens', 'train_batch_size', 'minibatch_size', 'num_epochs', 'lr', 'entropy_coeff', 'model_fcnet_activation', 'use_kl_loss']
std_cols = [c for c in priority if c in changing_cols]
other_cols = [c for c in changing_cols if c not in priority]
final_cols = ['Date', 'Checkpoint'] + std_cols + other_cols + ['Peak_Mean', 'Peak_Max', 'Peak_Min']

df_final = df[final_cols].sort_values(by='Date')

# Clean column names
clean_cols = [c.replace('model_', '').replace('fcnet_', '').replace('train_', '') for c in final_cols]
df_final.columns = clean_cols

print(df_final.to_markdown(index=False))
