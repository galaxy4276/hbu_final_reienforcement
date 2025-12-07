
import pandas as pd

files = [
    "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_d9bbc_00000_0_2025-12-03_16-22-39/progress.csv",
    "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_0b810_00000_0_2025-12-07_02-21-47/progress.csv",
    "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_f04ec_00000_0_2025-12-06_19-33-00/progress.csv"
]

for f in files:
    print(f"--- Analyzing {f.split('/')[-2]} ---")
    try:
        df = pd.read_csv(f)
        print("Columns:", df.columns.tolist())
        
        # Identify reward column
        candidates = ['env_runners/episode_reward_mean', 'episode_reward_mean', 'env_runners/episode_return_mean', 'episode_return_mean']
        col = None
        for c in candidates:
            if c in df.columns:
                col = c
                break
        
        if col:
            print(f"Using reward column: {col}")
            print(f"Max Reward: {df[col].max()}")
            print(f"Mean Reward (Last 10 avg): {df[col].tail(10).mean()}")
            print(f"Final Reward: {df[col].iloc[-1]}")
            print(f"Total Iterations: {len(df)}")
            
            # Print the first few non-nan values to see start
            print(f"Start Reward: {df[col].dropna().iloc[0]}")
        else:
            print("No reward column found.")
            
    except Exception as e:
        print(f"Error: {e}")
    print("\n")
