
import json
import pandas as pd
import os
import numpy as np
from datetime import datetime
import datetime as dt_module

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
                
                mtime = os.path.getmtime(os.path.join(root, "params.json"))
                dt_obj = datetime.fromtimestamp(mtime)
                if dt_obj.year < 2025 or dt_obj.month < 11: continue
                
                model = params.get('model', {})
                exp_data['Network Architecture'] = str(model.get('fcnet_hiddens'))
                exp_data['Batch Size'] = int(params.get('train_batch_size', 0))
                exp_data['MiniBatch Size'] = int(params.get('minibatch_size', 0))
                exp_data['Epochs'] = int(params.get('num_epochs', 10))
                
                lr = params.get('lr')
                if isinstance(lr, list):
                    start = lr[0][1]
                    end = lr[-1][1]
                    start_s = "3e-4" if start == 0.0003 else str(start)
                    exp_data['Learning Rate'] = f"{start_s} -> {end}"
                else:
                    lr_s = "3e-4" if lr == 0.0003 else str(lr)
                    exp_data['Learning Rate'] = f"{lr_s}"
                    
                exp_data['Entropy Coefficient'] = params.get('entropy_coeff')
                
                # Init & Activ
                init = model.get('post_fcnet_weights_initializer')
                init_conf = model.get('post_fcnet_weights_initializer_config', {})
                if init and "ortho" in init:
                    exp_data['Initialization'] = "Orthogonal (Gain 0.01)"
                else:
                    exp_data['Initialization'] = "Default"
                
                act = model.get('fcnet_activation', 'tanh')
                if act == 'Tanh': act = 'tanh'
                exp_data['Activation'] = act
                
                # 2. METRICS (Mean, Min, Max, Steps, Time)
                res_path = os.path.join(root, "result.json")
                
                p_mean = -np.inf
                p_max = -np.inf
                p_min = -np.inf
                steps_to_peak = 0
                time_total = 0 # This will be max time seen
                
                if os.path.exists(res_path):
                    with open(res_path, 'r') as rf:
                        for line in rf:
                            try:
                                rec = json.loads(line)
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
                                
                                # Steps
                                cur_steps = get_v(rec, ['timesteps_total', 'num_env_steps_sampled_lifetime'])
                                # Time
                                cur_time = get_v(rec, ['time_total_s'])
                                
                                if cur_time and cur_time > time_total:
                                    time_total = cur_time

                                # Capture Peak Mean snapshot
                                if cur_mean is not None and cur_mean > p_mean:
                                    p_mean = cur_mean
                                    if cur_max is not None: p_max = cur_max
                                    if cur_min is not None: p_min = cur_min
                                    if cur_steps is not None: steps_to_peak = cur_steps
                            except: pass
                            
                if p_mean != -np.inf:
                    exp_data['Peak Mean'] = f"{p_mean:.1f}"
                    exp_data['Peak Min'] = f"{p_min:.1f}"
                    exp_data['Peak Max'] = f"{p_max:.1f}"
                    # Format Steps: e.g. 2.0M or 500k
                    if steps_to_peak > 1000000:
                        exp_data['Steps to Peak'] = f"{steps_to_peak/1000000:.2f}M"
                    else:
                        exp_data['Steps to Peak'] = f"{int(steps_to_peak)}"
                    
                    # Format Time: HH:MM
                    if time_total > 0:
                        m, s = divmod(time_total, 60)
                        h, m = divmod(m, 60)
                        exp_data['Total Time'] = f"{int(h)}h {int(m)}m"
                    else:
                        exp_data['Total Time'] = "-"
                else:
                    exp_data['Peak Mean'] = "-"
                    exp_data['Steps to Peak'] = "-"
                    exp_data['Total Time'] = "-"
            
                experiments.append(exp_data)
            except: pass

df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    # Cols
    cols = ['Date', 'ID', 'Network Architecture', 'Batch Size', 'MiniBatch Size', 'Epochs', 'Learning Rate', 'Entropy Coefficient', 'Initialization', 'Activation', 'Peak Mean', 'Peak Min', 'Peak Max', 'Steps to Peak', 'Total Time']
    print(df[cols].to_markdown(index=False))
else:
    print("None")
