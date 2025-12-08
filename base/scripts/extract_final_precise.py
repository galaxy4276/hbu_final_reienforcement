
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
                
                # Checkpoint & Date
                parts = folder.split('_')
                if len(parts) > 2:
                    exp_data['ID'] = parts[2]
                    date_str = parts[-2] + " " + parts[-1].replace('-', ':')
                    exp_data['Date'] = date_str[5:16]
                else:
                    exp_data['ID'] = folder[:5]
                    exp_data['Date'] = "Unknown"
                
                # Timestamp filter (Dec 2025)
                mtime = os.path.getmtime(os.path.join(root, "params.json"))
                dt = datetime.fromtimestamp(mtime)
                if dt.year < 2025 or dt.month < 11: continue # allow late Nov
                
                model = params.get('model', {})
                
                # Network: Exact list
                exp_data['Network Architecture'] = str(model.get('fcnet_hiddens'))
                
                # Batch: Explicit Int
                exp_data['Batch Size'] = int(params.get('train_batch_size', 0))
                exp_data['MiniBatch Size'] = int(params.get('minibatch_size', 0))
                exp_data['Epochs'] = int(params.get('num_epochs', 10))
                
                # LR: Explicit string
                lr = params.get('lr')
                if isinstance(lr, list):
                    start = lr[0][1]
                    end = lr[-1][1]
                    # "3e-4 -> 0.0"
                    exp_data['Learning Rate'] = f"{start} -> {end}"
                else:
                    exp_data['Learning Rate'] = f"{lr}"
                    
                exp_data['Entropy Coefficient'] = params.get('entropy_coeff')
                exp_data['KL Loss'] = params.get('use_kl_loss', True)
                exp_data['Activation'] = model.get('fcnet_activation')
                
                # Init
                init = model.get('post_fcnet_weights_initializer')
                init_conf = model.get('post_fcnet_weights_initializer_config', {})
                if init:
                    if "ortho" in init:
                        gain = init_conf.get('gain', '1.0')
                        exp_data['Initialization'] = f"Orthogonal (Gain {gain})"
                    else:
                        exp_data['Initialization'] = init
                else:
                    exp_data['Initialization'] = "Default (Xavier/Kaiming)"
                
                # Other varying
                gamma = params.get('gamma', 0.99)
                lam = params.get('lambda', 1.0) # default? usually 1.0 or 0.95
                clip = params.get('grad_clip', None)
                exp_data['Gamma'] = gamma
                exp_data['Lambda'] = lam
                exp_data['Grad Clip'] = clip

                # 2. METRICS (Scan result.json for peaks)
                res_path = os.path.join(root, "result.json")
                
                # Trackers
                # We want "Peak Mean", and the corresponding Max/Min from that iteration? 
                # OR "Peak Max" seen ever? Usually "Peak Mean" is the performance metric.
                # User asked "Mean Max, Min값에대한 최대값" -> Max of Mean, Max of Max, Max of Min?
                # Usually standard table is: Best Mean, and (Max/Min at that point) or (Best Max ever).
                # I will calculate Global Max for each metric.
                
                p_mean = -np.inf
                p_max = -np.inf
                p_min = -np.inf
                
                valid_metric = False
                
                if os.path.exists(res_path):
                    with open(res_path, 'r') as rf:
                        for line in rf:
                            try:
                                rec = json.loads(line)
                                # Deep find
                                def get_v(d, k_list):
                                    for k in k_list:
                                        if k in d and isinstance(d[k], (int, float)): return d[k]
                                    for v in d.values():
                                        if isinstance(v, dict):
                                            res = get_v(v, k_list)
                                            if res is not None: return res
                                    return None

                                cur_mean = get_v(rec, ['episode_return_mean', 'episode_reward_mean'])
                                cur_max = get_v(rec, ['episode_return_max', 'episode_reward_max'])
                                cur_min = get_v(rec, ['episode_return_min', 'episode_reward_min'])
                                
                                if cur_mean is not None:
                                    valid_metric = True
                                    if cur_mean > p_mean: p_mean = cur_mean
                                if cur_max is not None:
                                    if cur_max > p_max: p_max = cur_max
                                if cur_min is not None:
                                    # For min, usually we want "Highest Min" (stability) or just statistics? 
                                    # "Max value for Min" -> Max(Min).
                                    if cur_min > p_min: p_min = cur_min
                            except: pass
                            
                if valid_metric:
                    exp_data['Peak Mean'] = f"{p_mean:.1f}"
                    exp_data['Peak Max'] = f"{p_max:.1f}"
                    exp_data['Peak Min'] = f"{p_min:.1f}"
                else:
                    exp_data['Peak Mean'] = "-"
                    exp_data['Peak Max'] = "-"
                    exp_data['Peak Min'] = "-"
            
                experiments.append(exp_data)
            except: pass

df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    # Cols
    cols = ['Date', 'ID', 'Network Architecture', 'Batch Size', 'MiniBatch Size', 'Epochs', 'Learning Rate', 'Entropy Coefficient', 'Initialization', 'Activation', 'Peak Mean', 'Peak Max', 'Peak Min']
    print(df[cols].to_markdown(index=False))
else:
    print("None")
