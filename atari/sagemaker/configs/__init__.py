"""
DQN Configuration for Atari Breakout SageMaker Training
"""

from .dqn_breakout_config import (
    register_custom_models,
    get_dqn_config,
    get_expert_training_config,
    get_full_config,
)

__all__ = [
    'register_custom_models',
    'get_dqn_config',
    'get_expert_training_config',
    'get_full_config',
]
