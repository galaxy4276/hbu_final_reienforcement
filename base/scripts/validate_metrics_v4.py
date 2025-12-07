
import pandas as pd
import subprocess
import os

base_dirs = {
    "Failure (PPO)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo",
    "Success (PPO v2)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2",
}

print(f"{'File':<50} | {'Return Mean':<15} | {'Reward Mean':<15} | {'Length':<10}")
print("-" * 100)

for cat, path in base_dirs.items():
    print(f"--- {cat} ---")
    try:
        files = subprocess.check_output(f"find {path} -name progress.csv", shell=True).decode().strip().split('\n')
        for f in files:
            if not f: continue
            try:
                df = pd.read_csv(f)
                if len(df) == 0: continue
                
                ret = "N/A"
                rew = "N/A"
                length = "N/A"
                
                # Check for return mean (Score)
                if 'env_runners/episode_return_mean' in df.columns:
                    ret = df['env_runners/episode_return_mean'].iloc[-1]
                elif 'episode_return_mean' in df.columns:
                    ret = df['episode_return_mean'].iloc[-1]
                
                # Check for reward mean
                if 'env_runners/episode_reward_mean' in df.columns:
                    rew = df['env_runners/episode_reward_mean'].iloc[-1]
                elif 'episode_reward_mean' in df.columns:
                    rew = df['episode_reward_mean'].iloc[-1]
                    
                # Check length
                if 'env_runners/episode_len_mean' in df.columns:
                    length = df['env_runners/episode_len_mean'].iloc[-1]
                elif 'episode_len_mean' in df.columns:
                    length = df['episode_len_mean'].iloc[-1]

                short_name = f.split('/')[-2]
                print(f"{short_name[:50]:<50} | {str(ret)[:15]:<15} | {str(rew)[:15]:<15} | {str(length)[:10]:<10}")
            except:
                pass
    except:
        pass
