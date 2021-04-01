import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels, sn=False):
        super().__init__()

        modules = []
        # implementing an architecture inspired by the paper referred to in the note book
        channels = [32, 64, 128, 256]
        
        for curr_channel in channels:
            if sn:
                modules.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, curr_channel, 5, stride=2, padding=2)))
            else:
                modules.append(nn.Conv2d(in_channels, curr_channel, 5, stride=2, padding=2))
            modules.append(nn.BatchNorm2d(curr_channel))
            modules.append(nn.LeakyReLU(0.2, True))
            in_channels = curr_channel

        # layer 5
        modules.append(nn.Conv2d(curr_channel, out_channels, 5, padding=2))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.LeakyReLU(0.2, True))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # implementing an architecture inspired by the paper referred to in the note book
        # layer 1 (decodes layer 4)
        channels = [256, 128, 64, 32]
        
        for curr_channel in channels:
            modules.append(nn.ConvTranspose2d(in_channels, curr_channel, 5, stride=2, padding=2, output_padding=1))
            modules.append(nn.BatchNorm2d(curr_channel))
            modules.append(nn.LeakyReLU(0.2, True))
            in_channels = curr_channel
        
        modules.append(nn.ConvTranspose2d(curr_channel, out_channels, 5, stride=1, padding=2))
        modules.append(nn.BatchNorm2d(out_channels))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))

