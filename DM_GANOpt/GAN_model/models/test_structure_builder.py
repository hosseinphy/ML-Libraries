import numpy as np

import os, sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from voxel_processing import (
    add_padding_to_voxel_grid,
    apply_marching_cubes,
    create_vtk_polydata,
    write_vtk_file,
    apply_laplacian_smoothing
)

from models.structure_assessor import Assessor


# build structrues with no overhangs and no resin traps
from islands_resintrap3 import get_resin_trap_mask


assessor = Assessor(occupancy_weight=10., resintrap_weight=100.)

#Mike's version
def create_voxel_structure(x_size, y_size, z_size, target_occupancy=0.5):
    structure = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Base layer (z = 0) — initialize a single voxel at center for testing
#    structure[x_size // 2, y_size // 2, 0] = True
    structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy
    for z in range(1, z_size):
        candidate_voxels = []
        # print(z)
        for x in range(1, x_size - 1):
            for y in range(1, y_size - 1):
                # Support check in 3x3 neighborhood at z-1
                support_area = structure[x - 1:x + 2, y - 1:y + 2, z - 1]
                if np.any(support_area):
                    candidate_voxels.append((x, y, z))

        num_to_occupy = int(len(candidate_voxels) * target_occupancy)

        if num_to_occupy > 0:
            chosen_indices = np.random.choice(len(candidate_voxels), num_to_occupy, replace=False)
            for idx in chosen_indices:
                x_, y_, z_ = candidate_voxels[idx]
                structure[x_, y_, z_] = True

    return structure



# My version
import numpy as np

def create_voxel_structure_no_overhangs(x_size, y_size, z_size, target_occupancy=0.5):
    structure = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Fill base layer randomly
    structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

    for z in range(1, z_size):
        for x in range(1, x_size - 1):
            for y in range(1, y_size - 1):
                # Check for support in 3x3 area directly below (z-1)
                support = structure[x - 1:x + 2, y - 1:y + 2, z - 1]
                if np.any(support):
                    # Randomly occupy if there's support below
                    structure[x, y, z] = np.random.rand() < target_occupancy

    return structure

def create_voxel_structure_no_overhangs_with_padding(x_size, y_size, z_size, target_occupancy=0.5):
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




def create_voxel_structure_no_overhangs_with_padding_extended(x_size, y_size, z_size, target_occupancy=0.3):
    structure = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Base layer — fill randomly to match occupancy
    structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

    for z in range(1, z_size):
        padded_prev = np.pad(structure[:, :, z - 1], pad_width=1, mode='constant', constant_values=0)
        candidates = []

        for x in range(x_size):
            for y in range(y_size):
                support_area = padded_prev[x:x + 3, y:y + 3]
                if np.any(support_area):
                    candidates.append((x, y))

        # Limit to match target occupancy
        num_voxels = int(target_occupancy * x_size * y_size)
        if len(candidates) > 0:
            selected_indices = np.random.choice(len(candidates), size=min(num_voxels, len(candidates)), replace=False)
            for idx in selected_indices:
                x, y = candidates[idx]
                structure[x, y, z] = True

    return structure



def create_voxel_structure_strict_no_overhangs(x_size, y_size, z_size, target_occupancy=0.3):
    structure = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Base layer — randomly fill to match occupancy
    num_voxels_per_layer = int(target_occupancy * x_size * y_size)
    structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

    for z in range(1, z_size):
        prev = structure[:, :, z - 1]
        padded_prev = np.pad(prev, pad_width=1, mode='constant', constant_values=0)

        valid_positions = []
        for x in range(x_size):
            for y in range(y_size):
                support_patch = padded_prev[x:x + 3, y:y + 3]
                if np.any(support_patch):
                    valid_positions.append((x, y))

        # Fill current layer uniformly from valid positions
        if not valid_positions:
            print(f"Stopped at z={z}: No supported positions.")
            break

        np.random.shuffle(valid_positions)
        for i in range(min(num_voxels_per_layer, len(valid_positions))):
            x, y = valid_positions[i]
            structure[x, y, z] = True

    return structure

import numpy as np
from scipy.ndimage import binary_opening

def create_voxel_structure_strict_no_overhangs_fixed(x_size, y_size, z_size, target_occupancy=0.3):
    structure = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Smoothed base to prevent isolated voxels
    base = np.random.rand(x_size, y_size) < target_occupancy
    structure[:, :, 0] = binary_opening(base, structure=np.ones((2,2)))

    num_voxels_per_layer = int(target_occupancy * x_size * y_size)

    for z in range(1, z_size):
        prev = structure[:, :, z - 1]
        padded_prev = np.pad(prev, pad_width=1, mode='constant', constant_values=0)
        valid_positions = []

        for x in range(x_size):
            for y in range(y_size):
                support_patch = padded_prev[x:x+3, y:y+3]
                center = support_patch[1, 1]
                edges = support_patch[0,1] or support_patch[1,0] or support_patch[1,2] or support_patch[2,1]
                if center or edges:
                    valid_positions.append((x, y))

        if not valid_positions:
            print(f"Stopped at z={z}: No supported positions.")
            break

        np.random.shuffle(valid_positions)
        for i in range(min(num_voxels_per_layer, len(valid_positions))):
            x, y = valid_positions[i]
            structure[x, y, z] = True

    return structure



import scipy.ndimage as ndimage

def remove_resin_traps(structure):
    """
    Remove enclosed voids (resin traps) by flood-filling from outside.
    Any void not connected to boundary is considered a trap and is filled.
    """
    voids = ~structure

    # Create seed mask from all 6 boundary faces
    seed_mask = np.zeros_like(voids, dtype=bool)
    seed_mask[0, :, :] = True
    seed_mask[-1, :, :] = True
    seed_mask[:, 0, :] = True
    seed_mask[:, -1, :] = True
    seed_mask[:, :, 0] = True
    seed_mask[:, :, -1] = True

    # Use binary_propagation to identify reachable air regions
    reachable_voids = ndimage.binary_propagation(seed_mask, mask=voids)

    # Enclosed voids = resin traps
    resin_traps = np.logical_and(voids, ~reachable_voids)

    # Fill resin traps with solid
    structure[resin_traps] = True

    return structure


def create_voxel_structure_no_overhangs_without_resin_traps(x_size, y_size, z_size, target_occupancy=0.5):
    structure = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Fill base layer randomly
    structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

    for z in range(1, z_size):
        padded_prev = np.pad(structure[:, :, z - 1], pad_width=1, mode='constant', constant_values=0)
        for x in range(x_size):
            for y in range(y_size):
                support_area = padded_prev[x:x + 3, y:y + 3]
                if np.any(support_area):
                    structure[x, y, z] = np.random.rand() < target_occupancy

    # Postprocess to remove resin traps
    structure = remove_resin_traps(structure)

    return structure




def create_voxel_structure_no_resintraps(x_size, y_size, z_size, target_occupancy=0.5, max_attempts=100):
    for attempt in range(max_attempts):
        structure = np.zeros((x_size, y_size, z_size), dtype=bool)

        # Fill base layer randomly
        structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

        for z in range(1, z_size):
            padded_prev = np.pad(structure[:, :, z - 1], pad_width=1, mode='constant', constant_values=0)
            for x in range(x_size):
                for y in range(y_size):
                    support_area = padded_prev[x:x + 3, y:y + 3]
                    if np.any(support_area):
                        structure[x, y, z] = np.random.rand() < target_occupancy

        # --- Check for resin traps ---
        resin_mask, num_trap_voxels = get_resin_trap_mask(structure.astype(np.uint8))
        if np.sum(resin_mask) == 0:
            print(f"✅ Valid structure generated without resin traps in {attempt+1} tries.")
            return structure.astype(np.uint8)

    raise RuntimeError("❌ Failed to generate resin-trap-free structure within max attempts.")


def create_voxel_structure_with_limit(x_size, y_size, z_size, target_occupancy=0.5, max_traps=20, max_attempts=200):
    for attempt in range(max_attempts):
        structure = np.zeros((x_size, y_size, z_size), dtype=bool)
        structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

        for z in range(1, z_size):
            padded_prev = np.pad(structure[:, :, z - 1], 1, mode='constant', constant_values=0)
            for x in range(x_size):
                for y in range(y_size):
                    if np.any(padded_prev[x:x + 3, y:y + 3]):
                        structure[x, y, z] = np.random.rand() < target_occupancy

        # Check resin traps
        resin_mask, _ = get_resin_trap_mask(structure.astype(np.uint8))
        num_traps = np.sum(resin_mask)

        if num_traps <= max_traps:
            print(f"✅ Generated structure with {int(num_traps)} resin traps.")
            return structure.astype(np.uint8)

    raise RuntimeError("❌ Failed to generate acceptable structure within limit.")



# # My version

# def create_voxel_structure_strict(x_size, y_size, z_size, target_occupancy=0.25,
#                                    support_threshold=0.6,
#                                    horizontal_threshold=0.3,
#                                    max_attempts=10):
#     """
#     Creates a voxel structure with strict overhang control.
#     Returns a structure with minimized unsupported voxels.
#     """
#     for attempt in range(max_attempts):
#         structure = np.zeros((x_size, y_size, z_size), dtype=bool)

#         # Base layer: random initialization
#         structure[:, :, 0] = np.random.rand(x_size, y_size) < target_occupancy

#         for z in range(1, z_size):
#             candidate_voxels = []

#             for x in range(2, x_size - 2):
#                 for y in range(2, y_size - 2):
#                     support_area = structure[x - 2:x + 3, y - 2:y + 3, z - 1]  # 5x5 kernel
#                     support_ratio = np.sum(support_area) / 25.0

#                     horizontal_area = structure[x - 1:x + 2, y - 1:y + 2, z]    # 3x3 same layer
#                     horizontal_ratio = np.sum(horizontal_area) / 9.0

#                     center_supported = structure[x, y, z - 1]

#                     if center_supported and support_ratio >= support_threshold and horizontal_ratio >= horizontal_threshold:
#                         candidate_voxels.append((x, y, z))

#             num_to_occupy = int(len(candidate_voxels) * target_occupancy)
#             if num_to_occupy > 0:
#                 chosen_indices = np.random.choice(len(candidate_voxels), num_to_occupy, replace=False)
#                 for idx in chosen_indices:
#                     x_, y_, z_ = candidate_voxels[idx]
#                     structure[x_, y_, z_] = True

#         # Quick structural sanity check (optional)
#         total_voxels = structure.sum()
#         if total_voxels > 50:  # or assessor(clean_tensor)['overhang'] < 0.02
#             return structure

#     print("Warning: Max attempts reached; returned structure may still have overhangs.")
#     return structure




import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
matplotlib.use('Agg')  # use non-interactive backend



def plot_voxels_3d(structure):
    x_size, y_size, z_size = structure.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(structure, facecolors='blue', edgecolors='k', linewidth=0.3, alpha=0.8)

    print("Number of occupied voxels:", np.sum(voxels))

    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_zlim(0, z_size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Layer Height)')
    ax.view_init(elev=30, azim=30)

    # plt.tight_layout()
    # plt.show()
    # Save to file instead of show
    plt.savefig("structure_3d.png")
    print("Plot saved as structure_3d.png")


def vtk_voxels_3d(structure, occ, ove, name):

    voxel_grid = structure
    # Ensure voxel grid is 3D
    if voxel_grid.ndim == 4 and voxel_grid.shape[0] == 1:
        voxel_grid = voxel_grid.squeeze(0)

    # Threshold the voxel grid to create a binary grid
    binary_voxel_grid = (voxel_grid > 0.5)#.cpu().numpy().astype(np.int32)

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
        write_vtk_file(smoothed_polydata, "smoothed_surface_mesh_real_{}_occ_{:.2f}_ove_{:.3f}.vtp".format(name, occ, ove))
    except ValueError as e:
        print(f"Marching Cubes algorithm failed: {e}")

# Example usage
#voxels = create_voxel_structure(80, 80, 80, target_occupancy=0.25)
if False:
    voxels = create_voxel_structure(32, 32, 32, target_occupancy=0.5).astype(np.float32)

    real_data_single = torch.tensor(
                voxels, dtype=torch.float32
            )
    clean_data = real_data_single.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1, 1)
    # real_scores = assessor(clean_data, target_occupancy=0.18)
    # real_occupancy = (clean_data > 0.5).float().mean().item()
    real_occupancy = clean_data.sum()/clean_data.numel()
    real_scores = assessor(clean_data, target_occupancy=real_occupancy)

    real_overhang = real_scores["overhang"]
    print("Real occupancy:", real_occupancy.item())
    print("Overhang score:", real_scores["overhang"].item())

    # voxels += 0.05 * np.random.randn(*voxels.shape)

    np.save("voxel_structure_test_overhangs_occ_{:.2f}_ove_{:.3f}.npy".format(real_occupancy, real_scores["overhang"].item()), voxels)
    print("Plotting now...")
    # plot_voxels_3d(voxels)
    vtk_voxels_3d(voxels, real_occupancy, real_scores["overhang"].item())

# voxels = create_voxel_structure_no_overhangs_with_padding(32, 32, 32, target_occupancy=0.5).astype(np.float32)
# voxels = create_voxel_structure_strict_no_overhangs_fixed(32, 32, 32, target_occupancy=0.3).astype(np.float32)
# voxels = create_voxel_structure_no_overhangs_without_resin_traps(32, 32, 32, target_occupancy=0.5).astype(np.float32)



voxels = create_voxel_structure_with_limit(32, 32, 32, target_occupancy=0.5).astype(np.float32)

real_data_single = torch.tensor(
                voxels, dtype=torch.float32
            )
clean_data = real_data_single.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1, 1)
# real_scores = assessor(clean_data, target_occupancy=0.18)
# real_occupancy = (clean_data > 0.5).float().mean().item()
real_occupancy = clean_data.sum()/clean_data.numel()
real_scores = assessor(clean_data, target_occupancy=real_occupancy)

real_overhang = real_scores["overhang"]
real_surface_score = real_scores["surface_score"]
print("Real occupancy:", real_occupancy.item())
print("Overhang score:", real_scores["overhang"].item())
print("Surface score:", real_scores["surface_score"].item())

# voxels += 0.05 * np.random.randn(*voxels.shape)

np.save("voxel_structure_test_withPadding_overhangs_occ_{:.2f}_ove_{:.3f}_surf_{:.3f}.npy".format(real_occupancy, real_scores["overhang"].item(), real_scores["surface_score"].item()), voxels)
print("Plotting now...")
# plot_voxels_3d(voxels)
vtk_voxels_3d(voxels, real_occupancy, real_scores["overhang"].item(), "withPadding")
