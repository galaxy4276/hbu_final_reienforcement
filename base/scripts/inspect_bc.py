
import pandas as pd
import os

path = "/home/kimjihun/ray_results/halfcheetah_offline_bc_legacy_pickle/MARWIL_HalfCheetah-v5_4531c_00000_0_2025-12-07_23-37-34/progress.csv"
try:
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    
    # Check specifically for evaluation columns
    eval_cols = [c for c in df.columns if 'evaluation' in c]
    print("Eval Columns:", eval_cols)
    
    if eval_cols:
        for c in eval_cols:
            print(f"{c}: {df[c].iloc[-1]}")
            
except Exception as e:
    print(e)
