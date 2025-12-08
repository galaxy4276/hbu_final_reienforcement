
import json
import pandas as pd
import os

checkpoints = {
    "Failure (d9bbc)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_d9bbc_00000_0_2025-12-03_16-22-39",
    "Success Blue (0b810)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_0b810_00000_0_2025-12-07_02-21-47",
    "Success Orange (f04ec)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_f04ec_00000_0_2025-12-06_19-33-00"
}

results = []

for name, path in checkpoints.items():
    csv_path = os.path.join(path, "progress.csv")
    metric_data = {}
    
    # Defaults
    metric_data['Mean'] = "N/A"
    metric_data['Min'] = "N/A"
    metric_data['Max'] = "N/A"

    try:
        df = pd.read_csv(csv_path)
        
        # Try finding return/reward cols
        for col in df.columns:
            if 'episode_return_mean' in col or 'episode_reward_mean' in col:
                metric_data['Mean'] = f"{df[col].iloc[-1]:.2f}"
            if 'episode_return_min' in col or 'episode_reward_min' in col:
                metric_data['Min'] = f"{df[col].iloc[-1]:.2f}"
            if 'episode_return_max' in col or 'episode_reward_max' in col:
                metric_data['Max'] = f"{df[col].iloc[-1]:.2f}"
    except:
        pass
        
    # Get Params (re-use logic)
    params_path = os.path.join(path, "params.json")
    param_data = {}
    try:
        with open(params_path, 'r') as f:
            data = json.load(f)
            model = data.get('model', {})
            param_data['Network'] = str(model.get('fcnet_hiddens'))
            param_data['Batch'] = data.get('train_batch_size')
            param_data['Epoch'] = data.get('num_epochs')
            param_data['Ent'] = data.get('entropy_coeff')
    except:
        pass

    results.append({**{'Model': name}, **param_data, **metric_data})

df_res = pd.DataFrame(results)
print(df_res.to_markdown(index=False))
