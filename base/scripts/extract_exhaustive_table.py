
import json
import pandas as pd
import os
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
                if dt.year < 2025 or dt.month < 12: continue
                if dt.day < 2: continue
                
                exp_data['Date'] = dt.strftime('%m-%d %H:%M')
                # Checkpoint Name (Hash)
                folder_name = root.split('/')[-1]
                # Format: PPO_HalfCheetah-v5_d9bbc_00000_0_...
                parts = folder_name.split('_')
                if len(parts) >= 4:
                    exp_data['Checkpoint'] = parts[2] # The hash 'd9bbc'
                else:
                    exp_data['Checkpoint'] = folder_name[:10]
                
                # Params
                with open(os.path.join(root, "params.json"), 'r') as f:
                    params = json.load(f)
                    
                model = params.get('model', {})
                exp_data['Network'] = str(model.get('fcnet_hiddens'))
                exp_data['Batch'] = params.get('train_batch_size')
                exp_data['MiniBatch'] = params.get('minibatch_size')
                exp_data['Epochs'] = params.get('num_epochs', 10)
                
                # LR
                lr = params.get('lr')
                if isinstance(lr, list): exp_data['LR'] = "Sched"
                else: exp_data['LR'] = f"{lr:.0e}"
                
                exp_data['Ent'] = params.get('entropy_coeff')
                exp_data['KL'] = params.get('use_kl_loss', True)
                exp_data['VF Share'] = model.get('vf_share_layers')
                exp_data['Grad Clip'] = params.get('grad_clip')
                exp_data['Activ'] = model.get('fcnet_activation')

                # Metrics
                csv_path = os.path.join(root, "progress.csv")
                metrics = "-"
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if len(df) > 0:
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
    # Reorder columns
    cols = ['Date', 'Checkpoint', 'Network', 'Batch', 'MiniBatch', 'Epochs', 'LR', 'Ent', 'KL', 'VF Share', 'Grad Clip', 'Activ', 'Result']
    print(df[cols].to_markdown(index=False))
else:
    print("None")
