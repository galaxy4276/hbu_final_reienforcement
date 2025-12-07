"""
Atari RL Utilities for SageMaker

Includes preprocessing functions and CNN model architectures.
"""

from .preprocess import (
    preprocess_observation,
    stack_frames,
    FrameStack,
    validate_observation_shape,
)

from .cnn_models import (
    AtariCNN,
    DuelingAtariCNN,
    BCCNN,
    get_model,
    AtariRLModule,
    DuelingAtariRLModule,
    get_rl_module_class,
)

__all__ = [
    'preprocess_observation',
    'stack_frames',
    'FrameStack',
    'validate_observation_shape',
    'AtariCNN',
    'DuelingAtariCNN',
    'BCCNN',
    'get_model',
    'AtariRLModule',
    'DuelingAtariRLModule',
    'get_rl_module_class',
]
