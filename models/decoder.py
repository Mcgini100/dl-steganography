import torch
import torch.nn as nn
from .common_blocks import ConvBlock

class Decoder(nn.Module):
    """
    Decoder Network: Extracts the embedded message from a (potentially noisy) image.
    Input: Container image (batch_size, 3, H, W)
    Output: Extracted message logits (batch_size, message_length)
    """
    def __init__(self, message_length, image_size, hidden_dim=64):
        super(Decoder, self).__init__()
        self.message_length = message_length
        self.image_size = image_size

        # Convolutional layers to extract features relevant to the message
        self.conv_layers = nn.Sequential(
            ConvBlock(3, hidden_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1), # Downsample
            ConvBlock(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            ConvBlock(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=2, padding=1), # Downsample
            ConvBlock(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=1, padding=1),
            ConvBlock(hidden_dim*4, hidden_dim*4, kernel_size=3, stride=2, padding=1), # Downsample
            # Add more blocks if needed
        )

        # Calculate the size after convolutions and downsampling
        # Example: If image_size=128, after 3 strides of 2 -> 128 / 2 / 2 / 2 = 16
        final_spatial_dim = image_size // (2**3) # Adjust based on number of stride=2 layers
        final_channels = hidden_dim * 4 # Adjust based on final ConvBlock output channels

        # Global Average Pooling or Flatten + Linear
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Output (batch_size, final_channels, 1, 1)

        # Fully connected layers to predict the message bits
        self.fc_layers = nn.Sequential(
            nn.Linear(final_channels, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, message_length)
            # No final activation here if using BCEWithLogitsLoss during training
            # Add nn.Sigmoid() or nn.Tanh() if using BCELoss or MSELoss respectively for the message
        )

    def forward(self, image):
        # 1. Pass through convolutional layers
        features = self.conv_layers(image)

        # 2. Pool features
        pooled_features = self.pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1) # Flatten: (batch_size, final_channels)

        # 3. Pass through fully connected layers to get message logits
        message_logits = self.fc_layers(pooled_features)

        return message_logits # Return logits