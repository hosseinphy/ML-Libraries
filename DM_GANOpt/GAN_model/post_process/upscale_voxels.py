# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 08:38:32 2025

@author: mgreenwo
"""

import os
import glob
import re
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path


def upscale_voxels(voxels, scale_factor=2):
    """Upscales the voxel grid using nearest neighbor to preserve structure."""
    return zoom(voxels.astype(float), scale_factor, order=0)

def infill_support(voxels):
    """Adds support voxels to fill diagonals and ensure connectivity."""
    padded = np.pad(voxels, 1, mode='constant')
    x_size, y_size, z_size = voxels.shape
    filled = padded.copy()

    # All 13 symmetric offset pairs in 3D
    directions = [
        ((-1, 0, 0), (1, 0, 0)),     # X axis
        ((0, -1, 0), (0, 1, 0)),     # Y axis
        ((0, 0, -1), (0, 0, 1)),     # Z axis
        ((-1, -1, 0), (1, 1, 0)),    # XY edge
        ((-1, 1, 0), (1, -1, 0)),    
        ((-1, 0, -1), (1, 0, 1)),    # XZ edge
        ((-1, 0, 1), (1, 0, -1)),    
        ((0, -1, -1), (0, 1, 1)),    # YZ edge
        ((0, -1, 1), (0, 1, -1)),
        ((-1, -1, -1), (1, 1, 1)),   # XYZ diagonal
        ((-1, -1, 1), (1, 1, -1)),
        ((-1, 1, -1), (1, -1, 1)),
        ((-1, 1, 1), (1, -1, -1)),
    ]

    for x in range(1, x_size+1):
        for y in range(1, y_size+1):
            for z in range(1, z_size+1):
                if not padded[x, y, z]:
                    for (dx1, dy1, dz1), (dx2, dy2, dz2) in directions:
                        if (padded[x+dx1, y+dy1, z+dz1] and
                            padded[x+dx2, y+dy2, z+dz2]):
                            filled[x, y, z] = 1
                            break

    return filled[1:-1, 1:-1, 1:-1] > 0.5


def create_upscaled_voxel(file_dir):

    scale_factor = 2

    file_dir = Path("./").resolve() #Path(file_dir).resolve()
    # file_dir = Path(file_dir).resolve()

    # for file in file_dir.glob("repaired_ohs_*.npy"):
    #     match = re.search(r"repaired_ohs_(\d+)\.npy", file.name)  # single underscore

    pattern_1  = ["repaired_ohs_*.npy"]
    pattern_2 = ["fake_voxel_batch_*.npy"]
    
    files_1 = [p for pat in pattern_1 for p in file_dir.glob(pat) ]
    files_2 = [p for pat in pattern_2 for p in file_dir.glob(pat) ]
    
    if files_1:
        pattern = pattern_1[0]
        re_pattern = r"repaired_ohs_(\d+)\.npy"  
    elif files_2:
        pattern = pattern_2[0]
        re_pattern = r"fake_voxel_batch_(\d+)\.npy"
    else:
        raise ValueError("The expected files not found.")
    
    #for file in file_dir.glob('fake_voxel_batch_*.npy'):
    for file in file_dir.glob(pattern):
        match = re.search(re_pattern, file.name)
        if match:
            iteration = int(match.group(1))

            input_path = file_dir / file  # pathlib way, cleaner
            output_path = file_dir / f"voxel_structure_upscaled_{iteration}.npy"

            try:
                voxels = np.load(input_path)
            except FileNotFoundError:
                print(f"File not found: {input_path}")
                continue  # avoid stopping the loop

            #print(f"\nProcessing {input_path}")
            #print("Original shape:", voxels.shape)

            upscaled = upscale_voxels(voxels, scale_factor)
            #print("Upscaled shape (before infill):", upscaled.shape)

            filled = infill_support(upscaled)
            #print("Final shape (after infill):", filled.shape)

            np.save(output_path, filled)
            #print(f"✅ Saved: {output_path}")

