
import pandas as pd

f = "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_d9bbc_00000_0_2025-12-03_16-22-39/progress.csv"

print(f"--- Analyzing d9bbc ---")
try:
    df = pd.read_csv(f)
    print("Columns:", df.columns.tolist())
    
    # Check return specifically
    if 'env_runners/episode_return_mean' in df.columns:
        col = 'env_runners/episode_return_mean'
    elif 'episode_return_mean' in df.columns:
        col = 'episode_return_mean'
    elif 'env_runners/episode_reward_mean' in df.columns:
        col = 'env_runners/episode_reward_mean'
        print("Warning: Using reward_mean, verify if this is return.")
    else:
        col = None

    if col:
        print(f"Using column: {col}")
        print(f"Max: {df[col].max()}")
        print(f"Mean (Last 10): {df[col].tail(10).mean()}")
        print(f"Final: {df[col].iloc[-1]}")
    else:
        print("No suitable column found.")
        
except Exception as e:
    print(f"Error: {e}")
