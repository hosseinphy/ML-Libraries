# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:45:44 2024

@author: sneve
"""

# voxel_processing.py

import open3d as o3d
import numpy as np
import vtk
from skimage import measure

def extract_voxel_indices(voxel_grid):
    voxels = voxel_grid.get_voxels()
    voxel_indices = [voxel.grid_index for voxel in voxels]
    return voxel_indices

def create_volume_array(voxel_indices, voxel_size, padding=1):
    if len(voxel_indices) == 0:
        return np.zeros((0, 0, 0), dtype=np.int8)  # Handle empty input

    # Ensure voxel_indices is a 2D array
    voxel_indices = np.array(voxel_indices)
    if voxel_indices.ndim == 1:
        voxel_indices = voxel_indices[np.newaxis, :]

    # Ensure voxel_indices are 3D indices
    voxel_indices = voxel_indices[:, :3]

    max_index = np.max(voxel_indices, axis=0)
    grid_shape = (max_index + 1 + padding * 2).astype(int)
    
    # Initialize a 3D volume array
    volume = np.zeros(grid_shape, dtype=np.int8)
    for idx in voxel_indices:
        padded_idx = (idx + padding).astype(int)
        volume[tuple(padded_idx)] = 1

    # Debugging: Check the content and shape of the volume
    print(f"Max index: {max_index}")
    print(f"Grid shape: {grid_shape}")
    print(f"Volume shape after creation: {volume.shape}")
    print(f"Volume non-zero elements: {np.count_nonzero(volume)}")
    return volume

def add_padding_to_voxel_grid(voxel_grid, padding=1):
    """
    Adds padding to the voxel grid to close off boundaries.

    Parameters:
    voxel_grid (numpy.ndarray): The input voxel grid.
    padding (int): The amount of padding to add.

    Returns:
    padded_voxel_grid (numpy.ndarray): The padded voxel grid.
    """
    padded_voxel_grid = np.pad(voxel_grid, pad_width=padding, mode='constant', constant_values=0)
    return padded_voxel_grid

def apply_marching_cubes(volume, voxel_size, padding=1):
    """
    Applies the Marching Cubes algorithm to create a surface mesh from the volume array
    and shifts the vertices back by the padding amount.

    Parameters:
    volume (numpy.ndarray): The 3D volume array.
    voxel_size (float): Size of each voxel.
    padding (int): The padding amount to shift back the coordinates.

    Returns:
    verts, faces, normals, values: Vertices, faces, normals, and values of the generated mesh.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume should be a 3D numpy array.")
    
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, spacing=(voxel_size, voxel_size, voxel_size))
    
    # Shift the vertices back by the padding amount
    verts -= padding * voxel_size
    
    return verts, faces, normals, values

def create_vtk_polydata(verts, faces):
    vtk_points = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    for vert in verts:
        vtk_points.InsertNextPoint(vert[0], vert[1], vert[2])
    for face in faces:
        vtk_cells.InsertNextCell(3)
        vtk_cells.InsertCellPoint(face[0])
        vtk_cells.InsertCellPoint(face[1])
        vtk_cells.InsertCellPoint(face[2])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)
    return polydata

def write_vtk_file(polydata, filename="surface_mesh.vtp"):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

def apply_laplacian_smoothing(polydata, iterations=20, relaxation_factor=0.1, feature_edge_smoothing=False, boundary_smoothing=True):
    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputData(polydata)
    smooth_filter.SetNumberOfIterations(iterations)
    smooth_filter.SetRelaxationFactor(relaxation_factor)
    if feature_edge_smoothing:
        smooth_filter.FeatureEdgeSmoothingOn()
    else:
        smooth_filter.FeatureEdgeSmoothingOff()
    if boundary_smoothing:
        smooth_filter.BoundarySmoothingOn()
    else:
        smooth_filter.BoundarySmoothingOff()
    smooth_filter.Update()
    return smooth_filter.GetOutput()
