"""
CNN model definitions for Atari Breakout with PyTorch.
Includes architectures for both DQN expert training and BC imitation learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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