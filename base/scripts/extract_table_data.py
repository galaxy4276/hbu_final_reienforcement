
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
    # 1. Get Params
    params_path = os.path.join(path, "params.json")
    param_data = {}
    try:
        with open(params_path, 'r') as f:
            data = json.load(f)
            model = data.get('model', {})
            param_data['Network'] = str(model.get('fcnet_hiddens'))
            param_data['Batch Size'] = data.get('train_batch_size')
            param_data['MiniBatch'] = data.get('minibatch_size')
            param_data['Epochs'] = data.get('num_epochs')
            param_data['LR'] = data.get('lr')
            if isinstance(param_data['LR'], list): param_data['LR'] = "Schedule"
            param_data['Entropy'] = data.get('entropy_coeff')
    except:
        pass

    # 2. Get Metrics (Last Row)
    csv_path = os.path.join(path, "progress.csv")
    metric_data = {}
    try:
        df = pd.read_csv(csv_path)
        # Handle column name variations
        def get_col(candidates):
            for c in candidates:
                if c in df.columns: return df[c].iloc[-1]
            return "N/A"

        metric_data['Mean'] = get_col(['env_runners/episode_reward_mean', 'episode_reward_mean'])
        metric_data['Min'] = get_col(['env_runners/episode_reward_min', 'episode_reward_min'])
        metric_data['Max'] = get_col(['env_runners/episode_reward_max', 'episode_reward_max'])
        
        # Format floats
        for k, v in metric_data.items():
            if isinstance(v, float):
                metric_data[k] = f"{v:.2f}"
                
    except:
        pass
        
    results.append({**{'Name': name}, **param_data, **metric_data})

# Print Markdown Table
df_res = pd.DataFrame(results)
print(df_res.to_markdown(index=False))
