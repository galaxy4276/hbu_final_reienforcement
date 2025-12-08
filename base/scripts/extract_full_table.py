
import json
import pandas as pd
import os
import glob
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
            try:
                # Meta
                mtime = os.path.getmtime(os.path.join(root, "params.json"))
                dt = datetime.fromtimestamp(mtime)
                if dt.year < 2025 or dt.month < 12: continue  # Filter old
                if dt.day < 2: continue # Filter before Dec 2
                
                exp_data['Date'] = dt.strftime('%m-%d %H:%M')
                exp_data['ID'] = root.split('_')[-2][:5] # Short hash
                
                # Params
                with open(os.path.join(root, "params.json"), 'r') as f:
                    params = json.load(f)
                    
                model = params.get('model', {})
                exp_data['Network'] = str(model.get('fcnet_hiddens'))
                exp_data['Batch'] = params.get('train_batch_size')
                exp_data['Epochs'] = params.get('num_epochs', 10) # Default 10 if missing
                exp_data['LR'] = params.get('lr')
                if isinstance(exp_data['LR'], list): exp_data['LR'] = "Sched"
                else: exp_data['LR'] = f"{exp_data['LR']:.0e}"
                
                exp_data['Ent'] = params.get('entropy_coeff')
                exp_data['KL'] = params.get('use_kl_loss', True)
                exp_data['Activ'] = model.get('fcnet_activation')

                # Metrics
                csv_path = os.path.join(root, "progress.csv")
                metrics = "N/A"
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if len(df) > 0:
                            # Try multiple column names
                            col = next((c for c in df.columns if 'episode_reward_mean' in c or 'episode_return_mean' in c), None)
                            if col:
                                val = df[col].iloc[-1]
                                if pd.notna(val):
                                    metrics = f"{val:.1f}"
                    except: pass
                
                exp_data['Result'] = metrics
                experiments.append(exp_data)
            except: pass

df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    df = df[['Date', 'ID', 'Network', 'Batch', 'Epochs', 'LR', 'Ent', 'KL', 'Activ', 'Result']]
    print(df.to_markdown(index=False))
else:
    print("None")
