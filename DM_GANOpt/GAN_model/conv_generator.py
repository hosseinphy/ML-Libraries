# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:33:39 2024

@author: sneve
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableSizeGenerator(nn.Module):
    def __init__(self, z_dim=100, base_dim=32, start_dim=4):
        super(VariableSizeGenerator, self).__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim  # Number of feature maps in the first layer
        self.start_dim = start_dim  # Initial spatial dimensions (e.g., 4x4x4)

        # Fully connected layer to project the noise vector into a dense representation
        self.fc = nn.Linear(z_dim, base_dim * start_dim * start_dim * start_dim)

        # Transposed convolutional layers to upsample to the desired size
        self.deconv1 = nn.ConvTranspose3d(base_dim, base_dim // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(base_dim // 2, base_dim // 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(base_dim // 4, base_dim // 8, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(base_dim // 8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z, target_size):
        """
        Forward pass of the generator.
        
        Parameters:
        z (torch.Tensor): Latent vector.
        target_size (tuple): Desired output size (depth, height, width).
        
        Returns:
        torch.Tensor: Generated voxel grid of the specified size.
        """
        batch_size = z.size(0)
        
        # Project and reshape the latent vector
        x = self.fc(z)
        x = x.view(batch_size, self.base_dim, self.start_dim, self.start_dim, self.start_dim)

        # Upsample using transposed convolutions
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        # Adaptive resizing to the target size
        x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=True)

        return x

# Example usage
device = 'cpu'
z_dim = 100
generator = VariableSizeGenerator(z_dim=z_dim, base_dim=32, start_dim=4).to(device)
z = torch.randn(1, z_dim, device=device)
target_size = (100, 100, 100)  # Example target size

# Generate voxel grid
voxel_grid = generator(z, target_size)
print(voxel_grid.shape)  # Should print: torch.Size([1, 1, 32, 32, 32])
print(voxel_grid)