import numpy as np

# import os, sys

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)

# from voxel_processing import (
#     add_padding_to_voxel_grid,
#     apply_marching_cubes,
#     create_vtk_polydata,
#     write_vtk_file,
#     apply_laplacian_smoothing
# )

# Mike's version
# def create_voxel_structure(x_size, y_size, z_size, target_occupancy=0.5):
#     structure = np.zeros((x_size, y_size, z_size), dtype=bool)

#     # Base layer (z = 0) — initialize a single voxel at center for testing
# #    structure[x_size // 2, y_size // 2, 0] = True
#     structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy
#     for z in range(1, z_size):
#         candidate_voxels = []
#         # print(z)
#         for x in range(1, x_size - 1):
#             for y in range(1, y_size - 1):
#                 # Support check in 3x3 neighborhood at z-1
#                 support_area = structure[x - 1:x + 2, y - 1:y + 2, z - 1]
#                 if np.any(support_area):
#                     candidate_voxels.append((x, y, z))

#         num_to_occupy = int(len(candidate_voxels) * target_occupancy)

#         if num_to_occupy > 0:
#             chosen_indices = np.random.choice(len(candidate_voxels), num_to_occupy, replace=False)
#             for idx in chosen_indices:
#                 x_, y_, z_ = candidate_voxels[idx]
#                 structure[x_, y_, z_] = True

#     return structure



# new version -- create padding at the boundaries to reduce unsupported voxels
def create_voxel_structure(x_size, y_size, z_size, target_occupancy=0.5):
    structure = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Fill base layer randomly
    structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

    for z in range(1, z_size):
        # Pad previous slice to avoid index errors at the edges
        padded_prev = np.pad(structure[:, :, z - 1], pad_width=1, mode='constant', constant_values=0)
        for x in range(x_size):
            for y in range(y_size):
                # Extract 3x3 support region centered at (x, y)
                support_area = padded_prev[x:x + 3, y:y + 3]
                if np.any(support_area):
                    structure[x, y, z] = np.random.rand() < target_occupancy

    return structure




# # build structrues with no overhangs and no resin traps
# from islands_resintrap3 import get_resin_trap_mask 


# def create_voxel_structure_no_resintraps(x_size, y_size, z_size, target_occupancy=0.5, max_attempts=100):
#     for attempt in range(max_attempts):
#         structure = np.zeros((x_size, y_size, z_size), dtype=bool)

#         # Fill base layer randomly
#         structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

#         for z in range(1, z_size):
#             padded_prev = np.pad(structure[:, :, z - 1], pad_width=1, mode='constant', constant_values=0)
#             for x in range(x_size):
#                 for y in range(y_size):
#                     support_area = padded_prev[x:x + 3, y:y + 3]
#                     if np.any(support_area):
#                         structure[x, y, z] = np.random.rand() < target_occupancy

#         # --- Check for resin traps ---
#         resin_mask, num_trap_voxels = get_resin_trap_mask(structure.astype(np.uint8))
#         if np.sum(resin_mask) == 0:
#             print(f"✅ Valid structure generated without resin traps in {attempt+1} tries.")
#             return structure.astype(np.uint8)

#     raise RuntimeError("❌ Failed to generate resin-trap-free structure within max attempts.")



# def create_voxel_structure(x_size, y_size, z_size, target_occupancy=0.5):
#     structure = np.zeros((x_size, y_size, z_size), dtype=bool)

#     # Base layer (z = 0) — initialize a single voxel at center for testing
# #    structure[x_size // 2, y_size // 2, 0] = True
#     structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy
#     for z in range(1, z_size):
#         candidate_voxels = []
#         # print(z)
#         for x in range(1, x_size - 1):
#             for y in range(1, y_size - 1):
#                 # Support check in 3x3 neighborhood at z-1
#                 support_area = structure[x - 1:x + 2, y - 1:y + 2, z - 1]
#                 if np.any(support_area):
#                     candidate_voxels.append((x, y, z))

#         num_to_occupy = int(len(candidate_voxels) * target_occupancy)

#         if num_to_occupy > 0:
#             chosen_indices = np.random.choice(len(candidate_voxels), num_to_occupy, replace=False)
#             for idx in chosen_indices:
#                 x_, y_, z_ = candidate_voxels[idx]
#                 structure[x_, y_, z_] = True

#     return structure






# # # My version
# def create_voxel_structure(x_size, y_size, z_size, target_occupancy=0.5):
#     structure = np.zeros((x_size, y_size, z_size), dtype=bool)
#     structure[:, :, 0] = np.random.rand(x_size, y_size) < (target_occupancy + 0.1)

#     for z in range(1, z_size):
#         candidate_voxels = []
#         for x in range(1, x_size - 1):
#             for y in range(1, y_size - 1):
#                 support_area = structure[x - 1:x + 2, y - 1:y + 2, z - 1]
#                 center_supported = support_area[1, 1]
#                 support_ratio = np.sum(support_area) / 9.0

#                 if center_supported and support_ratio >= 0.4:
#                     candidate_voxels.append((x, y, z))

#         num_to_occupy = int(len(candidate_voxels) * target_occupancy)
#         if num_to_occupy > 0:
#             chosen_indices = np.random.choice(len(candidate_voxels), num_to_occupy, replace=False)
#             for idx in chosen_indices:
#                 x_, y_, z_ = candidate_voxels[idx]
#                 structure[x_, y_, z_] = True

#     return structure


# def create_voxel_structure(x_size, y_size, z_size, target_occupancy=0.5):
#     structure = np.zeros((x_size, y_size, z_size), dtype=bool)

#     # Base layer (z = 0) — initialize a single voxel at center for testing
# #    structure[x_size // 2, y_size // 2, 0] = True
#     structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy
#     for z in range(1, z_size):
#         candidate_voxels = []
#         # print(z)
#         for x in range(1, x_size - 1):
#             for y in range(1, y_size - 1):
#                 # Support check in 3x3 neighborhood at z-1
#                 support_area = structure[x - 1:x + 2, y - 1:y + 2, z - 1]
#                 if np.any(support_area):
#                     candidate_voxels.append((x, y, z))

#         num_to_occupy = int(len(candidate_voxels) * target_occupancy)

#         if num_to_occupy > 0:
#             chosen_indices = np.random.choice(len(candidate_voxels), num_to_occupy, replace=False)
#             for idx in chosen_indices:
#                 x_, y_, z_ = candidate_voxels[idx]
#                 structure[x_, y_, z_] = True



# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# matplotlib.use('Agg')  # use non-interactive backend



# def plot_voxels_3d(structure):
#     x_size, y_size, z_size = structure.shape

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.voxels(structure, facecolors='blue', edgecolors='k', linewidth=0.3, alpha=0.8)

#     print("Number of occupied voxels:", np.sum(voxels))

#     ax.set_xlim(0, x_size)
#     ax.set_ylim(0, y_size)
#     ax.set_zlim(0, z_size)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z (Layer Height)')
#     ax.view_init(elev=30, azim=30)

#     # plt.tight_layout()
#     # plt.show()
#     # Save to file instead of show
#     plt.savefig("structure_3d.png")
#     print("Plot saved as structure_3d.png")


# def vtk_voxels_3d(structure):

#     voxel_grid = structure
#     # Ensure voxel grid is 3D
#     if voxel_grid.ndim == 4 and voxel_grid.shape[0] == 1:
#         voxel_grid = voxel_grid.squeeze(0)

#     # Threshold the voxel grid to create a binary grid
#     binary_voxel_grid = (voxel_grid > 0.5)#.cpu().numpy().astype(np.int32)

#     # Add padding to the binary voxel grid
#     padded_voxel_grid = add_padding_to_voxel_grid(binary_voxel_grid, padding=1)

#     # Ensure padded voxel grid is 3D
#     if padded_voxel_grid.ndim != 3:
#         padded_voxel_grid = padded_voxel_grid.squeeze()

#     # Apply Marching Cubes to create a surface mesh and shift vertices back
#     try:
#         verts, faces, normals, values = apply_marching_cubes(padded_voxel_grid, voxel_size=1.0, padding=1)

#         # Create VTK polydata
#         polydata = create_vtk_polydata(verts, faces)

#         # Apply Laplacian smoothing
#         smoothed_polydata = apply_laplacian_smoothing(polydata, iterations=20, relaxation_factor=0.1, feature_edge_smoothing=False, boundary_smoothing=True)

#         # Write the VTK file
#         write_vtk_file(smoothed_polydata, f"smoothed_surface_mesh_real.vtp")
#     except ValueError as e:
#         print(f"Marching Cubes algorithm failed: {e}")

# # Example usage
# #voxels = create_voxel_structure(80, 80, 80, target_occupancy=0.25)
# voxels = create_voxel_structure(32, 32, 32, target_occupancy=0.5)
# # np.save("voxel_structure.npy", voxels)
# print("Plotting now...")
# # plot_voxels_3d(voxels)
# vtk_voxels_3d(voxels)