import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from functools import partial
from . import get_sigmas
from .layers import *
from .normalization import get_normalization


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, config, stride=1):
        """
        A simple ResNet block with two convolutional layers and a skip connection.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolution. Used to downsample spatial dimensions.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection to match dimensions when stride > 1
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.act = get_act(config)

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)


class SimpleResNet(nn.Module):
    def __init__(self, config):
        """
        A simple ResNet model to map an image condition to a conditioning vector.
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            conditioning_dim (int): Size of the output conditioning vector.
        """
        super(SimpleResNet, self).__init__()
        self.config = config
        in_channels = config.data.meas_channels
        conditioning_dim = config.model.condition_dim
        self.act = get_act(config)

        self.initial_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global average pooling and fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, conditioning_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        Creates a ResNet layer with a specified number of blocks.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blocks (int): Number of ResNet blocks in the layer.
            stride (int): Stride for the first block.
        """
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, self.config, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, self.config, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn(self.initial_conv(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to a vector
        x = self.fc(x)  # Map to conditioning vector
        return x


class CINLayer(nn.Module):
    def __init__(self, hidden_dim, condition_dim):
        super(CINLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.gamma = nn.Linear(self.condition_dim, self.hidden_dim)
        self.beta = nn.Linear(self.condition_dim, self.hidden_dim)

    def forward(self, x, c):
        # Compute mean and std for normalization
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-5
        normalized_x = (x - mean) / std

        # Compute gamma and beta
        gamma = self.gamma(c)
        beta = self.beta(c)

        # Apply CIN
        return gamma * normalized_x + beta


class CINMLP(nn.Module):
    def __init__(self, config):
        super(CINMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.cin_layers = nn.ModuleList()
        self.act = get_act(config)
        self.register_buffer("sigmas", get_sigmas(config))
        self.config = config

        self.resnet_cond = SimpleResNet(config)

        # Input layer
        self.layers.append(nn.Linear(config.data.size, config.model.hidden_dims[0]))
        self.cin_layers.append(
            CINLayer(config.model.hidden_dims[0], config.model.condition_dim)
        )

        # Hidden layers
        for i in range(1, len(config.model.hidden_dims)):
            self.layers.append(
                nn.Linear(config.model.hidden_dims[i - 1], config.model.hidden_dims[i])
            )
            self.cin_layers.append(
                CINLayer(config.model.hidden_dims[i], config.model.condition_dim)
            )

        # Output layer
        self.output_layer = nn.Linear(config.model.hidden_dims[-1], config.data.size)

    def forward(self, x, y, s):
        h = x
        c = self.resnet_cond(y)

        for layer, cin in zip(self.layers, self.cin_layers):
            h = layer(h)
            h = cin(h, c)
            h = self.act(h)  # Activation function

        output = self.output_layer(h)

        used_sigmas = self.sigmas[s].view(x.shape[0], *([1] * len(output.shape[1:])))

        output = output / used_sigmas

        return output
