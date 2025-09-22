# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:14:36 2024

@author: sneve
"""
# gan_generator.py

import vtk  # Import VTK first to avoid conflicts
import torch
import torch.nn as nn
import numpy as np
from voxel_processing import (
    create_volume_array,
    apply_marching_cubes,
    create_vtk_polydata,
    write_vtk_file,
    apply_laplacian_smoothing
)

class Generator(nn.Module):
    def __init__(self, z_dim=100, voxel_dim=32):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.voxel_dim = voxel_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, voxel_dim * voxel_dim * voxel_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, 1, self.voxel_dim, self.voxel_dim, self.voxel_dim)

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

# Set device
device = 'cpu'

# Generator parameters
z_dim = 100
voxel_dim = 32
batch_size = 1  # Generate one voxel structure for simplicity

# Initialize the generator
generator = Generator(z_dim=z_dim, voxel_dim=voxel_dim).to(device)

# Generate random noise
noise = get_noise(batch_size, z_dim, device)

# Generate voxel structures
generated_voxels = generator(noise)

# Select one voxel structure for visualization
voxel_structure = generated_voxels[0].squeeze().cpu().detach().numpy()

# Create a volume array
voxel_indices = np.argwhere(voxel_structure > 0.5)  # Adjust threshold if necessary

volume = create_volume_array(voxel_indices, voxel_size=1.0)

# Apply Marching Cubes to create a surface mesh
verts, faces, normals, values = apply_marching_cubes(volume, voxel_size=1.0)

# Create VTK polydata
polydata = create_vtk_polydata(verts, faces)

# Apply Laplacian smoothing
smoothed_polydata = apply_laplacian_smoothing(polydata, iterations=20, relaxation_factor=0.1, feature_edge_smoothing=False, boundary_smoothing=True)

# Write the VTK file
write_vtk_file(smoothed_polydata, "smoothed_surface_mesh.vtp")

