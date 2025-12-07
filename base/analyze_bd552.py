
import pandas as pd

f = "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_bd552_00000_0_2025-12-06_10-41-52/progress.csv"
try:
    df = pd.read_csv(f)
    print("Columns:", df.columns.tolist())
    
    col = None
    if 'env_runners/episode_reward_mean' in df.columns:
        col = 'env_runners/episode_reward_mean'
    elif 'episode_reward_mean' in df.columns:
        col = 'episode_reward_mean'
        
    if col:
        print(f"Stats for {col}:")
        print(f"Max: {df[col].max()}")
        print(f"Mean: {df[col].mean()}")
        print(f"Std: {df[col].std()}")
        print("Last 20:")
        print(df[col].tail(20).to_string())
    else:
        print("No reward column found")
        
except Exception as e:
    print(e)
