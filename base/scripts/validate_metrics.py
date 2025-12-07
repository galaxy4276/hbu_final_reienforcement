
import pandas as pd
import os
import glob

base_dirs = {
    "Failure (PPO)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo",
    "Success (PPO v2)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2",
    "Offline BC": "/home/kimjihun/ray_results/halfcheetah_offline_bc_legacy_pickle"
}

print(f"{'Experiment':<50} | {'Mean Reward':<15} | {'Max Reward':<15} | {'Ep Len':<10}")
print("-" * 100)

for cat, path in base_dirs.items():
    print(f"--- {cat} ---")
    csv_files = glob.glob(os.path.join(path, "**", "progress.csv"), recursive=True)
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0: continue
            
            # Check for standard reward columns
            reward_col = 'env_runners/episode_reward_mean'
            if reward_col not in df.columns:
                reward_col = 'episode_reward_mean' # Old RLlib format
            
            if reward_col in df.columns:
                final_r = df[reward_col].iloc[-1]
                max_r = df[reward_col].max()
                
                # Check length
                len_col = 'env_runners/episode_len_mean'
                if len_col not in df.columns:
                     len_col = 'episode_len_mean'
                
                final_l = df[len_col].iloc[-1] if len_col in df.columns else "N/A"
                
                # Evaluation for BC
                eval_r = "N/A"
                for c in df.columns:
                    if 'evaluation' in c and 'reward_mean' in c:
                        eval_r = df[c].iloc[-1]
                        break
                
                # Clean up path for display
                short_name = f.split('/')[-2]
                
                print(f"{short_name:<50} | {final_r:<15.2f} | {max_r:<15.2f} | {final_l:<10}")
                
                if eval_r != "N/A":
                     print(f"   -> Eval Reward: {eval_r}")

        except Exception as e:
            pass
            # print(f"Error reading {f}: {e}")
