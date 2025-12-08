
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
                
                # Checkpoint ID
                folder = root.split('/')[-1]
                parts = folder.split('_')
                if len(parts) > 2:
                    exp_data['Checkpoint'] = parts[2]
                else:
                    exp_data['Checkpoint'] = folder[:5]
                
                # Params from params.json
                with open(os.path.join(root, "params.json"), 'r') as f:
                    params = json.load(f)
                
                model = params.get('model', {})
                exp_data['Network'] = str(model.get('fcnet_hiddens'))
                exp_data['Batch'] = params.get('train_batch_size')
                exp_data['MiniBatch'] = params.get('minibatch_size')
                exp_data['Epochs'] = params.get('num_epochs', 10)
                
                # LR
                lr = params.get('lr')
                if isinstance(lr, list):
                    start = lr[0][1]
                    end = lr[-1][1]
                    exp_data['LR'] = f"Sched({start}→{end})"
                else:
                    exp_data['LR'] = f"Fixed({lr})"
                    
                exp_data['Entropy'] = params.get('entropy_coeff')
                exp_data['KL'] = params.get('use_kl_loss', True)
                exp_data['Activ'] = model.get('fcnet_activation')
                
                # Initializer
                init = model.get('post_fcnet_weights_initializer')
                init_conf = model.get('post_fcnet_weights_initializer_config', {})
                if init:
                    exp_data['Init'] = f"Ortho({init_conf.get('gain', '')})" if "ortho" in init else init
                else:
                    exp_data['Init'] = "Default"
                    
                # Metrics from result.json (last line)
                res_path = os.path.join(root, "result.json")
                metrics = "-"
                if os.path.exists(res_path):
                    with open(res_path, 'r') as rf:
                        lines = rf.readlines()
                        if lines:
                            last_res = json.loads(lines[-1])
                            # Nested key search
                            def find_key(d, key_part):
                                for k, v in d.items():
                                    if key_part in k: return v
                                    if isinstance(v, dict):
                                        ret = find_key(v, key_part)
                                        if ret is not None: return ret
                                return None
                            
                            val = find_key(last_res, "episode_return_mean")
                            if val is not None:
                                metrics = f"{val:.1f}"
                            else:
                                # Fallback check for 'reward_mean'
                                val = find_key(last_res, "episode_reward_mean")
                                if val is not None:
                                     metrics = f"{val:.1f}"
                                
                exp_data['Result'] = metrics
                experiments.append(exp_data)
            except Exception as e:
                pass

df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    cols = ['Date', 'Checkpoint', 'Network', 'Batch', 'MiniBatch', 'Epochs', 'LR', 'Entropy', 'Init', 'Activ', 'Result']
    print(df[cols].to_markdown(index=False))
else:
    print("None")
