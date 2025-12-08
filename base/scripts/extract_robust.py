
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
            folder = root.split('/')[-1]
            try:
                # Basic info
                parts = folder.split('_')
                if len(parts) > 2:
                    exp_data['Checkpoint'] = parts[2]
                    date_str = parts[-2] + " " + parts[-1].replace('-', ':') 
                    exp_data['Date'] = date_str[5:16]
                else:
                    exp_data['Checkpoint'] = folder[:5]
                    exp_data['Date'] = "Unknown"

                # Params
                with open(os.path.join(root, "params.json"), 'r') as f:
                    params = json.load(f)
                
                model = params.get('model', {})
                exp_data['Network'] = str(model.get('fcnet_hiddens'))
                exp_data['Batch'] = params.get('train_batch_size')
                exp_data['MiniBatch'] = params.get('minibatch_size')
                exp_data['Epochs'] = params.get('num_epochs', 10)
                
                lr = params.get('lr')
                if isinstance(lr, list):
                    start = lr[0][1]
                    end = lr[-1][1]
                    exp_data['LR'] = f"Sched({start}→{end})"
                else:
                    exp_data['LR'] = f"Fixed({lr})"
                    
                exp_data['Entropy'] = params.get('entropy_coeff')
                
                # Init
                init = model.get('post_fcnet_weights_initializer')
                init_conf = model.get('post_fcnet_weights_initializer_config', {})
                if init:
                    exp_data['Init'] = f"Ortho({init_conf.get('gain', '')})" if "ortho" in init else init
                else:
                    exp_data['Init'] = "Default"
                    
                exp_data['Activ'] = model.get('fcnet_activation')

                # Metrics
                res_path = os.path.join(root, "result.json")
                if os.path.exists(res_path):
                    with open(res_path, 'r') as rf:
                        lines = rf.readlines()
                        if lines:
                            last_res = json.loads(lines[-1])
                            
                            def find_key(d, key_part):
                                # First pass: exact match
                                for k, v in d.items():
                                    if key_part == k and isinstance(v, (int, float)): 
                                        return v
                                # Second pass: substring match for scalars
                                for k, v in d.items():
                                    if key_part in k and isinstance(v, (int, float)):
                                        return v
                                # Third pass: recursive
                                for k, v in d.items():
                                    if isinstance(v, dict):
                                        ret = find_key(v, key_part)
                                        if ret is not None: return ret
                                return None
                            
                            val = find_key(last_res, "episode_return_mean")
                            if val is None: val = find_key(last_res, "episode_reward_mean")
                            
                            if val is not None:
                                exp_data['Result'] = f"{val:.1f}"
                            else:
                                exp_data['Result'] = "NoMetric"
                        else:
                            exp_data['Result'] = "EmptyRes"
                else:
                    exp_data['Result'] = "NoFile"

                experiments.append(exp_data)
            except Exception as e:
                # print(f"Error {folder}: {e}")
                pass

df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    cols = ['Date', 'Checkpoint', 'Network', 'Batch', 'MiniBatch', 'Epochs', 'LR', 'Entropy', 'Init', 'Activ', 'Result']
    print(df[cols].to_markdown(index=False))
else:
    print("None")
