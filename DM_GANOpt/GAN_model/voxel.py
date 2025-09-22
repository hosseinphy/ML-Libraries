# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:45:44 2024

@author: sneve
"""

import open3d as o3d
import numpy as np
import random
import vtk
from vtk.util import numpy_support
from skimage import measure

def create_voxel_grid(voxel_size=0.01, width=1.0, height=1.0, depth=1.0, occupancy_rate=0.4):
    """
    Creates a voxel grid with a given occupancy rate.

    Parameters:
    voxel_size (float): Size of each voxel.
    width (float): Width of the grid.
    height (float): Height of the grid.
    depth (float): Depth of the grid.
    occupancy_rate (float): Fraction of voxels that are occupied.

    Returns:
    voxel_grid (open3d.geometry.VoxelGrid): The generated voxel grid.
    """
    total_voxels = int((width / voxel_size) * (height / voxel_size) * (depth / voxel_size))
    num_occupied_voxels = int(occupancy_rate * total_voxels)

    all_voxel_indices = [(x, y, z) for x in range(int(width / voxel_size))
                         for y in range(int(height / voxel_size))
                         for z in range(int(depth / voxel_size))]

    occupied_voxels_indices = random.sample(all_voxel_indices, num_occupied_voxels)
    points = np.array(occupied_voxels_indices) * voxel_size

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def extract_voxel_indices(voxel_grid):
    """
    Extracts the indices of occupied voxels from the voxel grid.

    Parameters:
    voxel_grid (open3d.geometry.VoxelGrid): The voxel grid.

    Returns:
    voxel_indices (list of tuples): List of indices of occupied voxels.
    """
    voxels = voxel_grid.get_voxels()
    voxel_indices = [voxel.grid_index for voxel in voxels]
    return voxel_indices

def create_volume_array(voxel_indices, voxel_size, padding=1):
    """
    Creates a padded 3D numpy array (volume) from the voxel indices.

    Parameters:
    voxel_indices (list of tuples): List of indices of occupied voxels.
    voxel_size (float): Size of each voxel.
    padding (int): Padding size to add around the volume.

    Returns:
    volume (numpy.ndarray): The padded 3D volume array.
    """
    max_index = np.max(voxel_indices, axis=0)
    grid_shape = (max_index + 1 + padding * 2).astype(int)
    volume = np.zeros(grid_shape, dtype=np.int8)
    for idx in voxel_indices:
        padded_idx = (idx + padding).astype(int)
        volume[tuple(padded_idx)] = 1
    return volume

def apply_marching_cubes(volume, voxel_size):
    """
    Applies the Marching Cubes algorithm to create a surface mesh from the volume array.

    Parameters:
    volume (numpy.ndarray): The 3D volume array.
    voxel_size (float): Size of each voxel.

    Returns:
    verts, faces, normals, values: Vertices, faces, normals, and values of the generated mesh.
    """
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, spacing=(voxel_size, voxel_size, voxel_size))
    return verts, faces, normals, values

def create_vtk_polydata(verts, faces):
    """
    Creates a VTK polydata object from the vertices and faces of the mesh.

    Parameters:
    verts (numpy.ndarray): Vertices of the mesh.
    faces (numpy.ndarray): Faces of the mesh.

    Returns:
    polydata (vtk.vtkPolyData): The generated VTK polydata object.
    """
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
    """
    Writes the polydata to a VTK file.

    Parameters:
    polydata (vtk.vtkPolyData): The polydata to write.
    filename (str): The name of the output VTK file.
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

def apply_laplacian_smoothing(polydata, iterations=20, relaxation_factor=0.1, feature_edge_smoothing=False, boundary_smoothing=True):
    """
    Applies Laplacian smoothing to the polydata.

    Parameters:
    polydata (vtk.vtkPolyData): The input polydata.
    iterations (int): Number of smoothing iterations.
    relaxation_factor (float): Relaxation factor for smoothing.
    feature_edge_smoothing (bool): Whether to smooth feature edges.
    boundary_smoothing (bool): Whether to smooth boundaries.

    Returns:
    smoothed_polydata (vtk.vtkPolyData): The smoothed polydata.
    """
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

def main():
    """
    Main function that ties everything together and executes the workflow.
    """
    voxel_size = 0.05
    width = 1.0
    height = 1.0
    depth = 1.0
    occupancy_rate = 0.3

    voxel_grid = create_voxel_grid(voxel_size, width, height, depth, occupancy_rate)
    voxel_indices = extract_voxel_indices(voxel_grid)
    volume = create_volume_array(voxel_indices, voxel_size)
    verts, faces, normals, values = apply_marching_cubes(volume, voxel_size)
    polydata = create_vtk_polydata(verts, faces)
    smoothed_polydata = apply_laplacian_smoothing(polydata, iterations=20, relaxation_factor=0.1, feature_edge_smoothing=False, boundary_smoothing=True)
    write_vtk_file(smoothed_polydata, "smoothed_surface_mesh.vtp")

if __name__ == "__main__":
    main()

