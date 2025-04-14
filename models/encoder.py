import torch
import torch.nn as nn
from .common_blocks import conv_block

class Encoder(nn.Module):
    """Encoder network to hide secret bits into a cover image."""
    def __init__(self, secret_len=256, img_channels=3, initial_filters=64):
        super(Encoder, self).__init__()
        self.secret_len = secret_len
        f = initial_filters

        # --- Prepare Secret Branch ---
        # Processes the secret bit tensor to match image feature dimensions
        # Output channels should allow concatenation with image features later
        self.secret_processor = nn.Sequential(
            nn.Linear(secret_len, f * 4 * 4), # Example: project to match a spatial size
            nn.ReLU(inplace=True)
            # Potentially add more layers or reshaping logic here
        )
        self.secret_target_channels = f # How many channels secret will occupy after processing

        # --- Image Processing Branch ---
        # Initial convolution
        self.conv_in = conv_block(img_channels, f) # e.g., (B, 3, H, W) -> (B, 64, H, W)

        # Main convolutional layers (adjust depth/complexity as needed)
        self.conv_layers = nn.Sequential(
            conv_block(f, f),
            conv_block(f, f * 2), # -> (B, 128, H, W)
            conv_block(f * 2, f * 4), # -> (B, 256, H, W)
        )

        # --- Combined Branch ---
        # Adjust channels based on image features + processed secret
        combined_channels = f * 4 + self.secret_target_channels
        self.conv_combined = nn.Sequential(
            conv_block(combined_channels, f * 2), # Process combined features
            conv_block(f * 2, f),
            # Final layer produces the residual (same channels as input image)
            # No activation or BN here, bias might be useful
            nn.Conv2d(f, img_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Output activation to constrain the residual (e.g., [-1, 1])
        self.output_activation = nn.Tanh()

    def forward(self, cover_image, secret_bits):
        """
        Args:
            cover_image (torch.Tensor): Batch of cover images (B, C, H, W), normalized [0, 1].
            secret_bits (torch.Tensor): Batch of secret bits (B, SecretLen), values 0. or 1.
        Returns:
            torch.Tensor: Batch of stego images (B, C, H, W), normalized [0, 1].
        """
        # Process cover image
        img_features = self.conv_in(cover_image)
        img_features = self.conv_layers(img_features) # (B, F*4, H, W)

        # Process secret bits
        # Assume secret_processor outputs features matching a certain layer size
        # Here, we assume H, W are large enough, and we process secret to match H/N, W/N
        # A simpler approach for fixed size: project secret and reshape
        b, _, h, w = img_features.shape
        processed_secret = self.secret_processor(secret_bits) # (B, F*4*4)
        # Reshape secret to be spatially compatible for concatenation
        # Example: Reshape to (B, F, 4, 4) and then upscale or tile
        # A common technique: treat secret as additional channels
        # Reshape secret to (B, F, 1, 1) and expand
        processed_secret = processed_secret.view(b, self.secret_target_channels, 4, 4) # Adjust numbers based on layer sizes
        # Upsample or tile the secret feature map to match img_features size
        processed_secret_expanded = nn.functional.interpolate(processed_secret, size=(h, w), mode='bilinear', align_corners=False)
        # Alternative: Simply expand dims: processed_secret.view(b, self.secret_target_channels, 1, 1).expand(-1, -1, h, w)

        # Concatenate image features and processed secret
        combined_features = torch.cat([img_features, processed_secret_expanded], dim=1) # Concat along channel dim

        # Process combined features to get residual
        residual = self.conv_combined(combined_features)
        residual = self.output_activation(residual) # Residual is now roughly [-1, 1]

        # --- Adjust residual scaling if needed ---
        # Maybe scale residual down to make changes smaller, e.g., residual * 0.1
        # residual = residual * 0.1

        # Add residual to cover image
        stego_image = cover_image + residual

        # Clip output to valid range [0, 1]
        stego_image_clipped = torch.clamp(stego_image, 0.0, 1.0)

        return stego_image_clipped