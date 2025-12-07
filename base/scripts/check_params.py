
import json
import os

files = {
    "d9bbc (Failure)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo/PPO_HalfCheetah-v5_d9bbc_00000_0_2025-12-03_16-22-39/params.json",
    "0b810 (Success Blue)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_0b810_00000_0_2025-12-07_02-21-47/params.json",
    "f04ec (Success Orange)": "/home/kimjihun/ray_results/halfcheetah_expert_ppo_v2/PPO_HalfCheetah-v5_f04ec_00000_0_2025-12-06_19-33-00/params.json"
}

for name, path in files.items():
    print(f"--- {name} ---")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Network
        model = data.get('model', {})
        print(f"Network: {model.get('fcnet_hiddens')}")
        print(f"Activation: {model.get('fcnet_activation')}")
        print(f"Share Layers: {model.get('vf_share_layers')}")
        
        # Training
        print(f"Batch Size: {data.get('train_batch_size')}")
        print(f"Minibatch: {data.get('minibatch_size')}")
        print(f"LR: {data.get('lr')}")
        print(f"Entropy: {data.get('entropy_coeff')}")
        print(f"KL Target: {data.get('kl_target')}")
        
    except Exception as e:
        print(f"Error: {e}")
    print("\n")
