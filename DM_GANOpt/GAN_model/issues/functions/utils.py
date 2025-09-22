# functions/utils.py
from __future__ import annotations

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk

# Matplotlib (needed by plot_voxel)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D proj; ok if unused

from pathlib import Path

# --- Ensure project root is on sys.path (DM_GANOpt/) ---
#PROJECT_ROOT = Path(__file__).resolve().parent.parent  # functions/ -> project root
#if str(PROJECT_ROOT) not in sys.path:
#    sys.path.insert(0, str(PROJECT_ROOT))


from GAN_model.voxel_processing import (
    add_padding_to_voxel_grid,
    apply_marching_cubes,
    create_vtk_polydata,
    write_vtk_file,
    apply_laplacian_smoothing,
)


def add_frame(grid: np.ndarray, width: int = 1) -> np.ndarray:
    grid = np.pad(grid, width, 'constant', constant_values=0)
    grid[0:width, :, :] = 1
    grid[-width:, :, :] = 1
    grid[:, :, 0:width] = 1
    grid[:, :, -width:] = 1
    return grid

def remove_frame(grid: np.ndarray) -> np.ndarray:
    return grid[1:-1, 1:-1, 1:-1]

def get_instance_noise(epoch: int, max_noise: float = 0.1, decay_epochs: int = 10000) -> float:
    return max_noise * max(0.0, 1.0 - epoch / decay_epochs)

# ----- helpers -----
def plot_voxel(voxel: np.ndarray, title: str | None = None) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel, facecolors='blue', edgecolors='k', linewidth=0.3, alpha=0.8)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_vtk(voxel: np.ndarray, dirpath: str | Path, epoch: int, label: str | None = None) -> Path:
    """Example: build a VTK surface and write it to disk."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    # Your pipeline (adjust to match your functions’ signatures)
    padded = add_padding_to_voxel_grid(voxel, pad=1)
    verts, faces = apply_marching_cubes(padded)  # or whatever your function returns
    poly = create_vtk_polydata(verts, faces)
    poly = apply_laplacian_smoothing(poly, iterations=30)

    suffix = f"_{label}" if label else ""
    out = dirpath / f"smoothed_surface_mesh_epoch_{epoch}{suffix}.vtp"
    write_vtk_file(poly, out)
    return out


def calc_occupancy(voxel):
    return voxel.float().sum() /voxel.numel()
