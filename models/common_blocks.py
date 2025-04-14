import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, activation=nn.ReLU(inplace=True)):
    """Basic convolutional block."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_bn)
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(activation)
    return nn.Sequential(*layers)