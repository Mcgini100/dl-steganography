import torch
import torch.nn as nn
from .common_blocks import conv_block

class Decoder(nn.Module):
    """Decoder network to extract secret bits from a stego image."""
    def __init__(self, secret_len=256, img_channels=3, initial_filters=64):
        super(Decoder, self).__init__()
        f = initial_filters

        self.conv_layers = nn.Sequential(
            conv_block(img_channels, f), # (B, 3, H, W) -> (B, 64, H, W)
            conv_block(f, f),
            conv_block(f, f * 2, stride=2), # Downsample H/2, W/2 -> (B, 128, H/2, W/2)
            conv_block(f * 2, f * 2),
            conv_block(f * 2, f * 4, stride=2), # Downsample H/4, W/4 -> (B, 256, H/4, W/4)
            conv_block(f * 4, f * 4),
            conv_block(f * 4, f * 8, stride=2), # Downsample H/8, W/8 -> (B, 512, H/8, W/8)
            # Add more layers if needed
        )

        # Adaptive pooling to handle variable input sizes to some extent, outputs fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # Output size 4x4

        # Fully connected layers to map features to secret length
        # Input features depend on last conv channels and pooling output size
        fc_input_features = f * 8 * 4 * 4
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_features, fc_input_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fc_input_features // 4, secret_len)
            # No activation here, raw logits are often used with BCEWithLogitsLoss
            # Or use Sigmoid here if using BCELoss
        )

        # Optional: Add Sigmoid if using BCELoss later
        self.output_activation = nn.Sigmoid()


    def forward(self, image):
        """
        Args:
            image (torch.Tensor): Batch of stego images (B, C, H, W), normalized [0, 1].
        Returns:
            torch.Tensor: Batch of decoded secret bits probabilities (B, SecretLen), values [0, 1].
        """
        features = self.conv_layers(image)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1) # Flatten features
        secret_logits = self.fc_layers(features)

        # Apply sigmoid to get probabilities [0, 1]
        secret_probs = self.output_activation(secret_logits)

        return secret_probs # Output probabilities