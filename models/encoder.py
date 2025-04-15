import torch
import torch.nn as nn
from .common_blocks import ConvBlock

class Encoder(nn.Module):
    """
    Encoder Network: Embeds a message into a cover image.
    Input: Cover image (batch_size, 3, H, W), Message (batch_size, message_length)
    Output: Embedded image (batch_size, 3, H, W)
    """
    def __init__(self, message_length, image_size, hidden_dim=64):
        super(Encoder, self).__init__()
        self.message_length = message_length
        self.image_size = image_size

        # Layer to process the message and expand it spatially
        # Output channels = number of channels to concatenate with image features
        self.message_processor = nn.Sequential(
            nn.Linear(message_length, hidden_dim * (image_size // 4) * (image_size // 4)), # Expand message
            nn.BatchNorm1d(hidden_dim * (image_size // 4) * (image_size // 4)),
            nn.ReLU(inplace=True)
        )
        self.message_reshape_channels = hidden_dim

        # Initial convolution layer for the image
        self.conv_img_initial = ConvBlock(3, hidden_dim, kernel_size=3, stride=1, padding=1)

        # Main convolutional layers processing combined image features and message
        # Input channels = hidden_dim (from image) + self.message_reshape_channels (from message)
        self.conv_layers = nn.Sequential(
            ConvBlock(hidden_dim + self.message_reshape_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            # Add more blocks if needed for complexity
        )

        # Final layer to produce the residual (changes to the image)
        # Output should match image channels (3 for RGB)
        self.final_conv = nn.Conv2d(hidden_dim, 3, kernel_size=1, stride=1, padding=0)

        # Tanh activation to keep output bounded (assuming input images are normalized to [-1, 1])
        self.output_activation = nn.Tanh() # Use Sigmoid if images are [0, 1]

    def forward(self, image, message):
        batch_size = image.size(0)

        # 1. Process the message
        processed_message = self.message_processor(message)
        # Reshape message to match spatial dimensions for concatenation
        # Target shape: (batch_size, self.message_reshape_channels, H/4, W/4)
        # We need to calculate H/4 and W/4 based on image_size. Assume square image for simplicity.
        spatial_dim = self.image_size // 4
        message_feature_map = processed_message.view(batch_size, self.message_reshape_channels, spatial_dim, spatial_dim)
        # Upsample message feature map to match image feature map size (H x W) after initial conv
        message_feature_map_upsampled = nn.functional.interpolate(message_feature_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)


        # 2. Process the image
        image_features = self.conv_img_initial(image)

        # 3. Concatenate image features and processed message features
        # Ensure message_feature_map_upsampled has the same spatial size as image_features
        # If conv_img_initial changes size, adjust upsampling target size accordingly
        combined_features = torch.cat([image_features, message_feature_map_upsampled], dim=1) # Concatenate along channel dim

        # 4. Pass through main convolutional layers
        refined_features = self.conv_layers(combined_features)

        # 5. Generate the residual (subtle changes)
        residual = self.final_conv(refined_features)

        # 6. Add residual to the original image
        # Ensure residual is within a small range if needed, or rely on loss function
        embedded_image_raw = image + residual

        # 7. Apply final activation to constrain output range (e.g., [-1, 1])
        embedded_image = self.output_activation(embedded_image_raw)

        # Optional: Clamp values to ensure they are strictly within the expected range
        # embedded_image = torch.clamp(embedded_image, -1.0, 1.0)

        return embedded_image