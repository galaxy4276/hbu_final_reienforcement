
import pandas as pd
import subprocess
import os

base_dirs = {
    "Failure (PPO)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo",
    "Success (PPO v2)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2",
}

print(f"{'Experiment':<50} | {'Mean Reward':<15} | {'Max Reward':<15} | {'Ep Len':<10}")
print("-" * 100)

for cat, path in base_dirs.items():
    print(f"--- {cat} ---")
    try:
        # distinct find command
        files = subprocess.check_output(f"find {path} -name progress.csv", shell=True).decode().strip().split('\n')
        
        for f in files:
            if not f: continue
            try:
                df = pd.read_csv(f)
                if len(df) == 0: continue
                
                # Try to find reward mean
                reward_col = 'env_runners/episode_reward_mean'
                if reward_col not in df.columns:
                    reward_col = 'episode_reward_mean'
                
                if reward_col in df.columns:
                    final_r = df[reward_col].iloc[-1]
                    max_r = df[reward_col].max()
                    
                    # Try to find lengths
                    len_col = 'env_runners/episode_len_mean'
                    if len_col not in df.columns:
                        len_col = 'episode_len_mean'
                    final_l = df[len_col].iloc[-1] if len_col in df.columns else "N/A"

                    # Get parent folder name
                    short_name = f.split('/')[-2]
                    print(f"{short_name[:50]:<50} | {final_r:<15.2f} | {max_r:<15.2f} | {final_l:<10}")
            except:
                pass
    except Exception as e:
        print(f"Error finding files in {path}: {e}")
