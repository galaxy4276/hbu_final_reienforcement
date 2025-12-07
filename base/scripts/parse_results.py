
import pandas as pd
import sys

files = {
    "Success (v2)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_a55e5_00000_0_2025-12-07_11-08-39/progress.csv",
    "Failure (d9bbc)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_d9bbc_00000_0_2025-12-03_16-22-39/progress.csv",
    "Offline BC": "/home/kimjihun/ray_results/halfcheetah_offline_bc_legacy_pickle/MARWIL_HalfCheetah-v5_4531c_00000_0_2025-12-07_23-37-34/progress.csv"
}

for name, path in files.items():
    try:
        df = pd.read_csv(path)
        print(f"--- {name} ---")
        print(f"Total Rows: {len(df)}")
        
        # Reward Mean
        if 'env_runners/episode_reward_mean' in df.columns:
            final_reward = df['env_runners/episode_reward_mean'].iloc[-1]
            max_reward = df['env_runners/episode_reward_mean'].max()
            print(f"Final Reward Mean: {final_reward}")
            print(f"Max Reward Mean: {max_reward}")
        
        # Evaluation Reward Mean (for Offline BC)
        eval_col = 'evaluation/env_runners/episode_reward_mean' 
        if eval_col in df.columns:
             print(f"Final Eval Reward Mean: {df[eval_col].iloc[-1]}")
        else:
            # Check for any column with 'evaluation' and 'reward'
            eval_cols = [c for c in df.columns if 'evaluation' in c and 'reward_mean' in c]
            if eval_cols:
                for c in eval_cols:
                    print(f"Final {c}: {df[c].iloc[-1]}")

        # Loss (for Offline BC)
        loss_cols = [c for c in df.columns if 'loss' in c and 'total' in c]
        if loss_cols:
             for c in loss_cols:
                 print(f"Final {c}: {df[c].iloc[-1]}")
                 
        print("\n")
    except Exception as e:
        print(f"Error reading {name}: {e}\n")
