
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
    # Find all subdirectories containing params.json
    for root, dirs, files in os.walk(base_path):
        if "params.json" in files:
            exp_data = {}
            try:
                # 1. Parse Params
                with open(os.path.join(root, "params.json"), 'r') as f:
                    params = json.load(f)
                    
                model = params.get('model', {})
                exp_data['Date'] = datetime.fromtimestamp(os.path.getmtime(os.path.join(root, "params.json"))).strftime('%Y-%m-%d %H:%M')
                exp_data['ID'] = root.split('_')[-2] # Extract hash/id from folder name
                exp_data['Folder'] = root.split('/')[-1]
                
                exp_data['Network'] = str(model.get('fcnet_hiddens'))
                exp_data['Batch'] = params.get('train_batch_size')
                exp_data['Epochs'] = params.get('num_epochs')
                exp_data['LR'] = params.get('lr')
                if isinstance(exp_data['LR'], list): exp_data['LR'] = "Schedule"
                exp_data['Kl_loss'] = params.get('use_kl_loss')
                exp_data['Entropy'] = params.get('entropy_coeff')
                
                # classify Phase
                if exp_data['Network'] == '[1024, 1024]':
                    exp_data['Phase'] = "Phase 1 (Large)"
                elif exp_data['Kl_loss'] == False:
                     exp_data['Phase'] = "Phase 2 (No KL)"
                elif exp_data['Batch'] == 65536:
                     exp_data['Phase'] = "Phase 3 (SOTA)"
                else:
                    exp_data['Phase'] = "Other"

                # 2. Parse Metrics
                csv_path = os.path.join(root, "progress.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        # Find mean col
                        mean_col = next((c for c in df.columns if 'episode_reward_mean' in c or 'episode_return_mean' in c), None)
                        max_col = next((c for c in df.columns if 'episode_reward_max' in c or 'episode_return_max' in c), None)
                        
                        if mean_col:
                            exp_data['Mean'] = df[mean_col].iloc[-1]
                            exp_data['Max'] = df[mean_col].max() # Use mean curve max, or max col? User usually wants curve max.
                                                                  # But column 'max' is episode max. Let's use Last Mean.
                            exp_data['Peak Mean'] = df[mean_col].max()
                        if max_col:
                            exp_data['Abs Max'] = df[max_col].max()
                    else:
                        exp_data['Mean'] = "N/A"
                else:
                     exp_data['Mean'] = "No CSV"

                experiments.append(exp_data)
            except Exception as e:
                # print(f"Skipping {root}: {e}")
                pass

# Sort by Date
df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    # Filter columns
    cols = ['Phase', 'ID', 'Network', 'Batch', 'Epochs', 'Kl_loss', 'Mean', 'Peak Mean']
    print(df[cols].to_markdown(index=False))
else:
    print("No experiments found.")
