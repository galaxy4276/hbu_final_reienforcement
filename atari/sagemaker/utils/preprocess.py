"""
Atari-specific utilities for visual preprocessing and model definitions.
Includes image preprocessing functions and CNN architectures for Breakout.
"""
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """
    Preprocess Atari observation: Convert RGB to grayscale and resize.

    Args:
        obs: Raw observation (210, 160, 3) RGB

    Returns:
        Processed observation (84, 84) grayscale normalized
    """
    # Convert RGB to grayscale
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    # Resize to 84x84 (standard Atari preprocessing)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized = resized / 255.0

    return normalized


def stack_frames(frame_buffer: list, new_frame: np.ndarray, stack_size: int = 4) -> np.ndarray:
    """
    Stack frames to create temporal information.

    Args:
        frame_buffer: List of previous frames
        new_frame: New preprocessed frame (84, 84)
        stack_size: Number of frames to stack

    Returns:
        Stacked frames (84, 84, stack_size)
    """
    frame_buffer.append(new_frame)

    # Keep only the last stack_size frames
    if len(frame_buffer) > stack_size:
        frame_buffer.pop(0)

    # If not enough frames yet, repeat the first frame
    while len(frame_buffer) < stack_size:
        frame_buffer.insert(0, frame_buffer[0])

    return np.stack(frame_buffer, axis=-1)


class FrameStack(gym.Wrapper):
    """Frame stacking wrapper for Atari environments."""

    def __init__(self, env, stack_size: int = 4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frame_buffer = []

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(84, 84, stack_size),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = preprocess_observation(obs)
        self.frame_buffer = [processed_obs] * self.stack_size
        stacked = np.stack(self.frame_buffer, axis=-1).astype(np.float32)
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = preprocess_observation(obs)
        stacked_obs = stack_frames(self.frame_buffer, processed_obs, self.stack_size)
        return stacked_obs.astype(np.float32), reward, terminated, truncated, info


def validate_observation_shape(obs: np.ndarray) -> bool:
    """
    Validate that observation has the correct shape for CNN.

    Args:
        obs: Observation to validate

    Returns:
        True if shape is correct, False otherwise
    """
    expected_shape = (84, 84, 4)  # Height, Width, Stack size
    return obs.shape == expected_shape