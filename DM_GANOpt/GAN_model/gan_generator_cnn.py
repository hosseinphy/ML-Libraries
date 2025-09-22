# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:14:36 2024

@author: sneve
"""
# gan_generator.py

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class SimpleGenerator(nn.Module):
    def __init__(self, z_dim=100, base_dim=64):
        super(SimpleGenerator, self).__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim

        self.fc = nn.Linear(z_dim, base_dim * 4 * 4 * 4)
        self.deconv1 = nn.ConvTranspose3d(base_dim, base_dim // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(base_dim // 2, base_dim // 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(base_dim // 4, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, self.base_dim, 4, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class Assessor(nn.Module):
    def __init__(self):
        super(Assessor, self).__init__()

    def forward(self, voxel_grid):
        valid_voxels = voxel_grid[:, :, :32, :, :]
        invalid_voxels = voxel_grid[:, :, 32:, :, :]
        
        # Calculate score while maintaining gradients
        valid_score = valid_voxels.sum()
        invalid_score = invalid_voxels.sum()
        
        score = valid_score - invalid_score
        return score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dim = 100
generator = SimpleGenerator(z_dim=z_dim, base_dim=64).to(device)
assessor = Assessor().to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
num_epochs = 50

for epoch in range(num_epochs):
    generator.train()
    optimizer_g.zero_grad()

    z = torch.randn(1, z_dim, device=device)
    generated_voxels = generator(z)
    
    # Debugging prints
    print(f"Generated voxels requires_grad: {generated_voxels.requires_grad}")
    print(f"Generated voxels grad_fn: {generated_voxels.grad_fn}")

    score = assessor(generated_voxels)
    
    # Debugging prints
    print(f"Score requires_grad: {score.requires_grad}")
    print(f"Score grad_fn: {score.grad_fn}")
    
    loss_g = -score  # Ensure the score is a scalar

    # Debugging prints
    print(f"Loss G requires_grad: {loss_g.requires_grad}")
    print(f"Loss G grad_fn: {loss_g.grad_fn}")

    loss_g.backward()
    optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss G: {loss_g.item():.4f}")

    if epoch % 10 == 0:
        voxel_grid = generated_voxels[0].detach().cpu().numpy()
        # Optionally, add visualization or saving code here
