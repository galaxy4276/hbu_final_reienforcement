"""
CNN model definitions for Atari Breakout with PyTorch.
Includes architectures for both DQN expert training and BC imitation learning.
Now supports both legacy ModelV2 and new RLModule APIs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

# RLlib RLModule imports
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_distributions import TorchCategorical


class AtariCNN(nn.Module):
    """
    Standard CNN architecture for Atari games based on DeepMind's Nature paper.

    Architecture:
    - Conv2d(32, 8x8, stride=4) + ReLU
    - Conv2d(64, 4x4, stride=2) + ReLU
    - Conv2d(64, 3x3, stride=1) + ReLU
    - Flatten
    - Linear(512) + ReLU
    - Output layer
    """

    def __init__(self, input_channels: int = 4, action_dim: int = 4):
        super(AtariCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: (4, 84, 84)
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: (32, 20, 20)

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: (64, 9, 9)

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            # Output: (64, 7, 7)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (batch_size, height, width, channels) - will be transposed

        Returns:
            Q-values for each action
        """
        # Handle both channel-first and channel-last formats
        if x.dim() == 4 and x.shape[-1] == 4:
            # Convert from (batch, height, width, channels) to (batch, channels, height, width)
            x = x.permute(0, 3, 1, 2)

        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class DuelingAtariCNN(nn.Module):
    """
    Dueling DQN architecture for Atari games.
    Separates value and advantage streams for better performance.
    """

    def __init__(self, input_channels: int = 4, action_dim: int = 4):
        super(DuelingAtariCNN, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input format
        if x.dim() == 4 and x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)

        # Extract shared features
        features = self.feature_extractor(x)

        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class BCCNN(nn.Module):
    """
    Behavior Cloning CNN model for offline learning.
    Similar architecture to AtariCNN but optimized for imitation learning.
    """

    def __init__(self, input_channels: int = 4, action_dim: int = 4):
        super(BCCNN, self).__init__()

        # Convolutional layers (same as AtariCNN)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization in BC
            nn.Linear(512, action_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input format
        if x.dim() == 4 and x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)

        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def get_model(model_type: str = "cnn", input_channels: int = 4, action_dim: int = 4) -> nn.Module:
    """
    Factory function to get the appropriate model.

    Args:
        model_type: Type of model ("cnn", "dueling", "bc")
        input_channels: Number of input channels (usually 4 for frame stacking)
        action_dim: Number of possible actions

    Returns:
        PyTorch model
    """
    models = {
        "cnn": AtariCNN,
        "dueling": DuelingAtariCNN,
        "bc": BCCNN
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](input_channels, action_dim)


class AtariRLModule(TorchRLModule):
    """
    RLModule implementation for Atari games using standard CNN architecture.
    Compatible with RLlib 2.50.1+ RLModule API.
    """

    def __init__(self, observation_space, action_space, inference_only=False, learner_only=False, model_config=None):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config or {}
        )

    @override
    def setup(self):
        """Initialize the CNN network layers."""
        obs_space = self.config.observation_space
        action_space = self.config.action_space

        # Determine input channels from observation space
        if hasattr(obs_space, 'shape') and len(obs_space.shape) == 3:
            # Assume (H, W, C) format, get channels from last dimension
            input_channels = obs_space.shape[-1] if obs_space.shape[-1] == 4 else 4
        else:
            input_channels = 4  # Default for frame-stacked Atari

        action_dim = action_space.n if hasattr(action_space, 'n') else 4

        # Store for potential use
        self.input_channels = input_channels
        self.action_dim = action_dim

        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def _forward_net(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN network."""
        # Handle both channel-first and channel-last formats
        if obs.dim() == 4 and obs.shape[-1] == 4:
            # Convert from (batch, height, width, channels) to (batch, channels, height, width)
            obs = obs.permute(0, 3, 1, 2)

        x = self.conv_layers(obs)
        x = self.fc_layers(x)
        return x

    @override
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for inference (action selection)."""
        obs = batch["obs"]
        q_values = self._forward_net(obs)

        return {
            "q_values": q_values,
            "action_dist_inputs": q_values,
        }

    @override
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for exploration (training)."""
        return self._forward_inference(batch)

    @override
    def get_inference_action_dist(self, inputs: torch.Tensor) -> TorchCategorical:
        """Create action distribution for inference."""
        # For DQN, we use categorical distribution over Q-values
        return TorchCategorical(logits=inputs)

    @override
    def get_train_action_dist(self, inputs: torch.Tensor) -> TorchCategorical:
        """Create action distribution for training."""
        return self.get_inference_action_dist(inputs)


class DuelingAtariRLModule(TorchRLModule):
    """
    RLModule implementation for Atari games using Dueling DQN architecture.
    Separates value and advantage streams for better performance.
    Compatible with RLlib 2.50.1+ RLModule API.
    """

    def __init__(self, observation_space, action_space, inference_only=False, learner_only=False, model_config=None):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config or {}
        )

    @override
    def setup(self):
        """Initialize the dueling CNN network layers."""
        obs_space = self.config.observation_space
        action_space = self.config.action_space

        # Determine input channels and action dimensions
        if hasattr(obs_space, 'shape') and len(obs_space.shape) == 3:
            input_channels = obs_space.shape[-1] if obs_space.shape[-1] == 4 else 4
        else:
            input_channels = 4

        action_dim = action_space.n if hasattr(action_space, 'n') else 4

        self.input_channels = input_channels
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def _forward_net(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dueling network."""
        # Handle input format
        if obs.dim() == 4 and obs.shape[-1] == 4:
            obs = obs.permute(0, 3, 1, 2)

        # Extract shared features
        features = self.feature_extractor(obs)

        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    @override
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for inference."""
        obs = batch["obs"]
        q_values = self._forward_net(obs)

        return {
            "q_values": q_values,
            "action_dist_inputs": q_values,
        }

    @override
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for exploration."""
        return self._forward_inference(batch)

    @override
    def get_inference_action_dist(self, inputs: torch.Tensor) -> TorchCategorical:
        """Create action distribution for inference."""
        return TorchCategorical(logits=inputs)

    @override
    def get_train_action_dist(self, inputs: torch.Tensor) -> TorchCategorical:
        """Create action distribution for training."""
        return self.get_inference_action_dist(inputs)


def get_rl_module_class(model_type: str = "cnn") -> type:
    """
    Factory function to get the appropriate RLModule class.

    Args:
        model_type: Type of model ("cnn" or "dueling")

    Returns:
        RLModule class
    """
    rl_modules = {
        "cnn": AtariRLModule,
        "dueling": DuelingAtariRLModule,
    }

    if model_type not in rl_modules:
        raise ValueError(f"Unknown RLModule type: {model_type}. Available: {list(rl_modules.keys())}")

    return rl_modules[model_type]


def test_model():
    """Test the model with dummy input."""
    # Test AtariCNN
    model = get_model("cnn", input_channels=4, action_dim=4)

    # Test with both input formats
    x1 = torch.randn(1, 4, 84, 84)  # Channel-first
    x2 = torch.randn(1, 84, 84, 4)  # Channel-last

    print("Testing AtariCNN:")
    print(f"Input shape (channel-first): {x1.shape}")
    output1 = model(x1)
    print(f"Output shape: {output1.shape}")

    print(f"Input shape (channel-last): {x2.shape}")
    output2 = model(x2)
    print(f"Output shape: {output2.shape}")

    # Test DuelingAtariCNN
    dueling_model = get_model("dueling", input_channels=4, action_dim=4)
    dueling_output = dueling_model(x1)
    print(f"Dueling CNN output shape: {dueling_output.shape}")

    print("All models working correctly!")


if __name__ == "__main__":
    test_model()