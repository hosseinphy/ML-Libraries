import os, sys
import torch
import numpy as np
import json
import vtk
import pandas as pd
from vtk.util.numpy_support import numpy_to_vtk


parent_dir = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(parent_dir)



def add_frame(grid, width=1):
    """Add solid top and bottom layer to grid, as well on two vertial sides.

    Increases size of grid by 1 for each dimension.
    """

    grid = np.pad(grid, width, 'constant', constant_values=0)

    # Bottom and Top
    grid[0:width, :, :] = 1
    grid[-width::, :, :] = 1

    # Sides
    grid[:, :, 0:width] = 1
    grid[:, :,-width::] = 1

    return grid

def remove_frame(grid):
    return grid[1:-1, 1:-1, 1:-1]



def get_instance_noise(epoch, max_noise=0.1, decay_epochs=10000):
    return max_noise * max(0.0, 1.0 - epoch / decay_epochs)


from voxel_processing import (
    add_padding_to_voxel_grid,
    apply_marching_cubes,
    create_vtk_polydata,
    write_vtk_file,
    apply_laplacian_smoothing
)


# functions 
def plot_voxel(voxel, title=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel, facecolors='blue', edgecolors='k', linewidth=0.3, alpha=0.8)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_vtk(voxel, dirpath, epoch, label=None):
    # with torch.no_grad():
    #     z = torch.randn(1, latent_dim).to(device)
    #     ema_fake = ema.ema_model(z)
    #     voxel_grid = ema_fake[0, 0]

    # Ensure voxel grid is 3D
    if voxel.ndim == 4 and voxel.shape[0] == 1:
        voxel = voxel.squeeze(0)


    if isinstance(voxel, torch.Tensor):
        voxel = voxel.detach().cpu().numpy()


    # Threshold the voxel grid to create a binary grid
    binary_voxel_grid = (voxel > 0.5).astype(np.int32)


    #binary_voxel_grid = (voxel > 0.5).cpu().numpy().astype(np.int32)

    # Add padding to the binary voxel grid
    padded_voxel_grid = add_padding_to_voxel_grid(binary_voxel_grid, padding=1)

    # Ensure padded voxel grid is 3D
    if padded_voxel_grid.ndim != 3:
        padded_voxel_grid = padded_voxel_grid.squeeze()

    # Apply Marching Cubes to create a surface mesh and shift vertices back
    try:
        verts, faces, normals, values = apply_marching_cubes(padded_voxel_grid, voxel_size=1.0, padding=1)

        # Create VTK polydata
        polydata = create_vtk_polydata(verts, faces)

        # Apply Laplacian smoothing
        smoothed_polydata = apply_laplacian_smoothing(polydata, iterations=20, relaxation_factor=0.1, feature_edge_smoothing=False, boundary_smoothing=True)

        # Write the VTK file
        vtk_path = os.path.join(dirpath, f"{label}_smoothed_surface_mesh_epoch_{epoch}.vtp")
        write_vtk_file(smoothed_polydata, vtk_path)
        # write_vtk_file(smoothed_polydata, f"smoothed_surface_mesh_epoch_{epoch}.vtp")
    except ValueError as e:
        print(f"Marching Cubes algorithm failed: {e}")



def calc_occupancy(voxel):
    return voxel.float().sum() /voxel.numel()
