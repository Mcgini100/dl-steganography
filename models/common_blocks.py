import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv -> BatchNorm (optional) -> Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True, activation=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

# Example of a slightly more complex block if needed
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, activation=None) # No activation before adding residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return nn.ReLU(inplace=True)(out) # Apply activation after adding residual