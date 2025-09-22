# -*- coding: utf-8 -*-
"""
Created on Fri May 07 14:22:00 2025

@author: hazizi
"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt


# # Disciminator 3D
# class Discriminator3D(nn.Module):
#     def __init__(self, base_channels=64):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv3d(1, base_channels, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU( 0.2, inplace=True),

#             nn.Conv3d( base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels*2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d( base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels*4),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d(base_channels*4, 1, kernel_size=4),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x).view(-1,1)
    

# IF BCEWithLogitsLoss is used for training, Sigmoid must not be used for the last layer.

# Disciminator 3D ver2.0 (for BCEWithLogitsLoss )
# class Discriminator3D(nn.Module):
#     def __init__(self, base_channels=64):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv3d(1, base_channels, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU( 0.2, inplace=True),

#             nn.Conv3d( base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels*2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d( base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels*4),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d(base_channels*4, 1, kernel_size=4),
#             # nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x).view(-1,1)
    

# Disciminator 3D ver3.0 (for BCEWithLogitsLoss and using spectral norm and remove batchnorm )
class Discriminator3D(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv3d(1, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(base_channels * 4, 1, kernel_size=4))
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)



# # test Discriminator
# disc  = Discriminator3D(base_channels=64)
# disc.eval()

# # fake structure
# x = torch.zeros(1, 1, 32, 32, 32)
# x[:, :, 10:20, 10:20, 10:20] = 1.0 # insert a cube


# with torch.no_grad():
#     output = disc(x)
#     print("Discriminator output:", output.item())


