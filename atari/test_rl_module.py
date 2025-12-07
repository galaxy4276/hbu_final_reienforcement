#!/usr/bin/env python
"""
Test script to verify RLModule implementation works correctly.
"""
import sys
import gymnasium as gym
import ale_py
import torch
import numpy as np

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent  # Go up to final_reinforcement
sys.path.insert(0, str(project_root))

from atari.utils.cnn_models import AtariRLModule, DuelingAtariRLModule, get_rl_module_class
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

def test_rl_modules():
    """Test RLModule implementations."""
    print("🧪 Testing RLModule implementations...")

    # Register ALE environments
    gym.register_envs(ale_py)

    # Create a temporary environment to get spaces
    env = gym.make("ALE/Breakout-v5")

    # Apply frame stacking wrapper
    from atari.utils.preprocess import FrameStack
    env = FrameStack(env, stack_size=4)

    obs_space = env.observation_space
    action_space = env.action_space

    print(f"  Observation space: {obs_space}")
    print(f"  Action space: {action_space}")

    # Test AtariRLModule
    print("\n🔹 Testing AtariRLModule...")
    try:
        rl_module = AtariRLModule(
            observation_space=obs_space,
            action_space=action_space,
            inference_only=False,
            learner_only=False,
            model_config={}
        )

        # Test forward pass with dummy data
        batch_size = 2
        dummy_obs = np.random.randint(0, 255, (batch_size, 84, 84, 4), dtype=np.uint8)
        batch = {"obs": dummy_obs}

        # Test inference
        with torch.no_grad():
            result = rl_module._forward_inference(batch)
            q_values = result["q_values"]

        print(f"  ✅ Inference successful! Output shape: {q_values.shape}")
        assert q_values.shape == (batch_size, action_space.n), f"Expected shape {(batch_size, action_space.n)}, got {q_values.shape}"

        # Test exploration
        with torch.no_grad():
            result_explore = rl_module._forward_exploration(batch)
            q_values_explore = result_explore["q_values"]

        print(f"  ✅ Exploration successful! Output shape: {q_values_explore.shape}")

        print("  ✅ AtariRLModule test passed!")

    except Exception as e:
        print(f"  ❌ AtariRLModule test failed: {e}")
        return False

    # Test DuelingAtariRLModule
    print("\n🔹 Testing DuelingAtariRLModule...")
    try:
        dueling_rl_module = DuelingAtariRLModule(
            observation_space=obs_space,
            action_space=action_space,
            inference_only=False,
            learner_only=False,
            model_config={}
        )

        # Test forward pass with dummy data
        batch_size = 2
        dummy_obs = np.random.randint(0, 255, (batch_size, 84, 84, 4), dtype=np.uint8)
        batch = {"obs": dummy_obs}

        # Test inference
        with torch.no_grad():
            result = dueling_rl_module._forward_inference(batch)
            q_values = result["q_values"]

        print(f"  ✅ Inference successful! Output shape: {q_values.shape}")
        assert q_values.shape == (batch_size, action_space.n), f"Expected shape {(batch_size, action_space.n)}, got {q_values.shape}"

        print("  ✅ DuelingAtariRLModule test passed!")

    except Exception as e:
        print(f"  ❌ DuelingAtariRLModule test failed: {e}")
        return False

    # Test factory function
    print("\n🔹 Testing get_rl_module_class function...")
    try:
        cnn_class = get_rl_module_class("cnn")
        dueling_class = get_rl_module_class("dueling")

        assert cnn_class == AtariRLModule, "CNN class mismatch"
        assert dueling_class == DuelingAtariRLModule, "Dueling class mismatch"

        print("  ✅ Factory function test passed!")

    except Exception as e:
        print(f"  ❌ Factory function test failed: {e}")
        return False

    env.close()
    return True

def test_config_integration():
    """Test configuration integration."""
    print("\n🔧 Testing configuration integration...")

    try:
        from atari.configs.dqn_breakout_config import get_dqn_config

        # Test new RLModule API config
        config_new = get_dqn_config(
            env_name="atari_breakout",
            model_type="dueling",
            use_new_api_stack=True
        )

        print("  ✅ RLModule API config created successfully")

        # Test legacy ModelV2 API config
        config_legacy = get_dqn_config(
            env_name="atari_breakout",
            model_type="dueling",
            use_new_api_stack=False
        )

        print("  ✅ Legacy ModelV2 API config created successfully")

        return True

    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting RLModule refactoring test...")

    success = True

    # Test RLModule implementations
    if not test_rl_modules():
        success = False

    # Test configuration integration
    if not test_config_integration():
        success = False

    print("\n" + "="*50)
    if success:
        print("🎉 ALL TESTS PASSED! RLModule refactoring is working correctly.")
        print("\nYou can now run the training script with:")
        print("  python atari_1_Training_DQN_expert.py --model-type dueling")
        print("  python atari_1_Training_DQN_expert.py --model-type dueling --use-legacy-api")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
        sys.exit(1)