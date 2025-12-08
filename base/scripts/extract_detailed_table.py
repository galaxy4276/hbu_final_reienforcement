
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
                exp_data['Checkpoint'] = root.split('/')[-1].split('_')[2] if len(root.split('/')[-1].split('_')) > 2 else root.split('/')[-1][:5]
                
                # Params
                with open(os.path.join(root, "params.json"), 'r') as f:
                    params = json.load(f)
                
                model = params.get('model', {})
                
                # Critical Params
                exp_data['Network'] = str(model.get('fcnet_hiddens'))
                exp_data['Batch'] = params.get('train_batch_size')
                exp_data['MiniBatch'] = params.get('minibatch_size')
                exp_data['Epochs'] = params.get('num_epochs', 10)
                
                # LR Detail
                lr = params.get('lr')
                if isinstance(lr, list): 
                    # Extract start/end LR from schedule [[0, 3e-4], [1e7, 0.0]]
                    try:
                        start_lr = lr[0][1]
                        end_lr = lr[-1][1]
                        exp_data['LR'] = f"Sched({start_lr}->{end_lr})"
                    except:
                         exp_data['LR'] = "Sched"
                else: 
                     exp_data['LR'] = f"{lr}"

                exp_data['Entropy'] = params.get('entropy_coeff')
                exp_data['KL Loss'] = params.get('use_kl_loss', True)
                exp_data['Grad Clip'] = params.get('grad_clip')
                exp_data['Activ'] = model.get('fcnet_activation')
                
                # Subtle Params
                exp_data['Gamma'] = params.get('gamma')
                exp_data['Lambda'] = params.get('lambda')
                
                # Init
                init = model.get('post_fcnet_weights_initializer')
                init_conf = model.get('post_fcnet_weights_initializer_config', {})
                if init:
                    exp_data['Init'] = f"{init}({init_conf.get('gain','')})"
                else:
                    exp_data['Init'] = "Default"

                experiments.append(exp_data)
            except: pass

df = pd.DataFrame(experiments)
if not df.empty:
    df = df.sort_values(by='Date')
    # Filter columns that are constant to avoid clutter, or keep them if important
    # Let's print all first to see
    print(df.to_markdown(index=False))
else:
    print("None")
