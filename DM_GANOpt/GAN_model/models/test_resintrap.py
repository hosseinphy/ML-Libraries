import torch
import torch.nn.functional as F
import sys, os
import numpy as np

from scipy.ndimage import label

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from voxel_processing import (
    add_padding_to_voxel_grid,
    apply_marching_cubes,
    create_vtk_polydata,
    write_vtk_file,
    apply_laplacian_smoothing
)

import vtk
from vtk.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt


# def plot_voxel(voxel, title=None):
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.voxels(voxel, facecolors='blue', edgecolors='k', linewidth=0.3, alpha=0.8)
#     if title:
#         ax.set_title(title)
#     plt.tight_layout()
#     plt.show()


def plot_voxel_np(voxel_np, title="Voxel"):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_np, facecolors='red', edgecolors='k', linewidth=0.2, alpha=0.9)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_enclosed_voids(trap_mask_np, title="Enclosed Voids (Resin Traps)"):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot only where the mask is non-zero (i.e., resin trap locations)
    x, y, z = np.nonzero(trap_mask_np)
    ax.scatter(x, y, z, c='red', marker='o', s=5, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()



# def resintrap_score_2d(voxel_grid):
#     device = voxel_grid.device
#     penalty = 0.0
#     B, C, D, H, W = voxel_grid.shape
#     kernel = torch.ones(1, 1, 3, 3).to(device)
#     padding = kernel.shape[-1] // 2

#     for z in range(D):
#         layer = voxel_grid[:, :, z, :, :]
#         inverted = 1.0 - layer
#         local_sum = F.conv2d(layer, kernel, padding=padding)
#         fully_enclosed = (local_sum == kernel.sum()).float()
#         penalty += (inverted * fully_enclosed).sum()

#     return penalty / (B * D * H * W)

def resintrap_score_2d(voxel_grid):
    B, C, D, H, W = voxel_grid.shape
    device = voxel_grid.device
    kernel2d = torch.ones(1, 1, 3, 3, device=device)
    padding = kernel2d.shape[-1] // 2
    penalty = 0.0
    trap_mask = torch.zeros_like(voxel_grid)

    for z in range(D):
        layer = voxel_grid[:, :, z, :, :]         # [B, 1, H, W]
        voids = 1.0 - layer
        local_sum = F.conv2d(layer, kernel2d, padding=padding)
        enclosed = (local_sum >= 8).float()       # Fully enclosed in 2D
        traps = voids * enclosed                  # Only apply to voids
        trap_mask[:, :, z, :, :] = traps
        penalty += traps.sum()

    score = penalty / (B * D * H * W)
    return score, trap_mask



def resintrap_score_3d(voxel_grid):
    """
    Detects 3D hard resin traps: voids (zeros) that are completely enclosed by solid (ones) in 3D.
    Args:
        voxel_grid: torch.Tensor of shape [B, 1, D, H, W], values in [0, 1]
    Returns:
        Scalar differentiable resin trap score (higher = worse)
    """
    B, C, D, H, W = voxel_grid.shape
    device = voxel_grid.device

    # Define a 3D kernel to check 3x3x3 neighborhood (excluding center if desired)
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    # kernel[0, 0, 1, 1, 1] = 0  # Optionally exclude center voxel

    # Identify candidate voids
    voids = 1.0 - voxel_grid  # 1 = void, 0 = solid

    # Count how many of a void's neighbors are solid
    neighbor_solid_count = F.conv3d(voxel_grid, kernel, padding=1)

    # Number of neighbors (26 if we exclude center, 27 if not)
    max_neighbors = kernel.sum() -1

    # Fully enclosed void: it's a void and surrounded by solids 
    # (greater sign is to account for to allow for some numerical tolerance or flexibility)
    enclosed_mask = (neighbor_solid_count >= max_neighbors).float()

    resin_traps = voids * enclosed_mask
    score = resin_traps.sum() / (B * D * H * W)
    # Normalize and return scalar score
    return score, resin_traps


def unified_resintrap_score(voxel_grid, alpha_2d=0.5, alpha_3d=0.5):
    score_2d, mask_2d = resintrap_score_2d(voxel_grid)
    score_3d, mask_3d = resintrap_score_3d(voxel_grid)

    combined_score = alpha_2d * score_2d + alpha_3d * score_3d
    combined_mask = ((alpha_2d * mask_2d + alpha_3d * mask_3d) > 0).float()

    return {
        "score_2d": score_2d,
        "score_3d": score_3d,
        "combined_score": combined_score,
        "mask_2d": mask_2d,
        "mask_3d": mask_3d,
        "combined_mask": combined_mask
    }



def diff_find_enclosed_voids(voxel_grid, seed_mask, iterations=20, threshold=0.5, steepness=10.0):
    B, C, D, H, W = voxel_grid.shape
    device = voxel_grid.device

    # Define a 3D kernel for 6-connectivity
    kernel = torch.tensor([[[[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]],
                            [[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]],
                            [[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]]]], dtype=torch.float32, device=device)
    kernel = kernel.unsqueeze(0)

    mask = seed_mask.clone()
    for _ in range(iterations):
        conv_result = F.conv3d(mask, kernel, padding=1)
        activated = torch.sigmoid(steepness * (conv_result - threshold))
        mask = torch.where(voxel_grid == 0, torch.max(mask, activated), mask)

    # Apply hard threshold to avoid partial gradients causing leaks
    binary_mask = (mask > 0.95).float()

    return binary_mask


def filter_small_clusters(mask_np, min_voxels=5):
    labeled, n = label(mask_np)
    for i in range(1, n + 1):
        if np.sum(labeled == i) < min_voxels:
            mask_np[labeled == i] = 0
    return mask_np

def resin_trap_score(voxel_grid, resintrap_weight=1.0, min_cluster_size=5):
    B, C, D, H, W = voxel_grid.shape

    # Seed from surface boundaries (6 sides)
    seed_mask = torch.zeros_like(voxel_grid)
    seed_mask[:, :, 5, 5, 5] = 1
    
    # seed_mask[:, :, 0, :, :] = 1
    # seed_mask[:, :, -1, :, :] = 1
    # seed_mask[:, :, :, 0, :] = 1
    # seed_mask[:, :, :, -1, :] = 1
    # seed_mask[:, :, :, :, 0] = 1
    # seed_mask[:, :, :, :, -1] = 1

    enclosed_voids_mask = diff_find_enclosed_voids(voxel_grid, seed_mask)

    # Remove tiny artifacts in mask (postprocessing for viz)
    mask_np = enclosed_voids_mask[0, 0].detach().cpu().numpy()
    mask_np_filtered = filter_small_clusters(mask_np.copy(), min_voxels=min_cluster_size)
    filtered_mask = torch.from_numpy(mask_np_filtered).to(voxel_grid.device).unsqueeze(0).unsqueeze(0)

    score = resintrap_weight * filtered_mask.sum() / (B * D * H * W)

    return {
        "rtp": score,
        "enclosed_voids": filtered_mask
    }



# def diff_find_enclosed_voids(voxel_grid, seed_mask, iterations=25, threshold=0.5, steepness=20.0):
#     B, C, D,  H, W = voxel_grid.shape

#     device = voxel_grid.device

#     # Define a 3D kernel for 6-connectivity
#     kernel = torch.tensor([[[[0, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 0]],
#                             [[0, 1, 0],
#                              [1, 0, 1],
#                              [0, 1, 0]],
#                             [[0, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 0]]]], dtype=torch.float32, device=device)
#     kernel = kernel.unsqueeze(0)

#     mask = seed_mask.clone()
#     for _ in range(iterations):
#          # Convolve the current mask
#         conv_result = F.conv3d(mask, kernel, padding=1)
#          # Apply sigmoid activation
#         activated = torch.sigmoid(steepness * (conv_result - threshold) )
        
#         # update mask

#         # mask = torch.max(mask, activated)
#         # # Ensure thre fill does not go to the solid regions
#         # mask = mask * (1-voxel_grid)
    
#         # The code block above may cause Leak to solid areas!!!!
#         mask = torch.where(voxel_grid == 0, torch.max(mask, activated), mask)

#     return mask


# def resin_trap_score(voxel_grid, resintrap_weight):
#     """
#     Penalize resin trap formation in the voxel grid.
#     """

#     B, C, D, H, W = voxel_grid.shape
    
#     seed_mask = torch.zeros_like(voxel_grid)
#     # Initiating flood-fill from a single voxel at (5,5,5) in every batch.
#     # If the outer voids are not connected to this point, entire regions may be 
#     # misclassified as resin traps, especially in larger grids or offset shapes.
#     #--------------------------------------------------------------------------#
#     seed_mask[:, :, 5, 5, 5] = 1
#     # seed_mask[:, :, 0, :, :] = 1
#     # seed_mask[:, :, -1, :, :] = 1
#     # seed_mask[:, :, :, 0, :] = 1
#     # seed_mask[:, :, :, -1, :] = 1
#     # seed_mask[:, :, :, :, 0] = 1
#     # seed_mask[:, :, :, :, -1] = 1


#     # Compute enclosed voids penalty -> resin traps
#     enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
#     rtp = resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

#     return {
#         "rtp": rtp,
#         "enclosed_voids": enclosed_voids
#     }



def refined_resintrap_2d(voxel_grid, iterations=10, threshold=4.0, steepness=50.0, solid_touch_kernel_size=5):
    """
    Differentiable 2D resin trap detection: excludes distant voids.
    Returns:
        score, trap_mask
    """
    B, C, D, H, W = voxel_grid.shape
    device = voxel_grid.device

    # 4-connected kernel
    kernel = torch.tensor([[[[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]]]], dtype=torch.float32, device=device)

    # kernel for detecting solid proximity
    solid_kernel = torch.ones((1, 1, solid_touch_kernel_size, solid_touch_kernel_size), device=device)

    trap_mask = torch.zeros_like(voxel_grid)
    total_voxels = 0.0

    for z in range(D):
        slice_ = voxel_grid[:, :, z, :, :]
        voids = 1.0 - slice_

        # Seed fill from edges
        seed_mask = torch.zeros_like(slice_)
        seed_mask[:, :, 0, :] = 1
        seed_mask[:, :, -1, :] = 1
        seed_mask[:, :, :, 0] = 1
        seed_mask[:, :, :, -1] = 1

        # Flood-fill reachable voids
        mask = seed_mask.clone()
        for _ in range(iterations):
            conv = F.conv2d(mask, kernel, padding=1)
            activated = torch.sigmoid(steepness * (conv - threshold))
            mask = torch.where(voids > 0, torch.max(mask, activated), mask)

        # Now: unreached voids are potentially enclosed
        unreached_voids = voids * (1 - mask)

        # Only keep unreached voids close to solid material
        solid_neighbors = F.conv2d(slice_, solid_kernel, padding=solid_touch_kernel_size // 2)
        solid_touching = (solid_neighbors > 0).float()
        solid_touching = solid_touching * unreached_voids

        trap_mask[:, :, z, :, :] = solid_touching
        total_voxels += solid_touching.sum()

    score = total_voxels / (B * D * H * W)
    return score, trap_mask




def write_to_vtk(voxel_grid):
    # Ensure voxel grid is 3D
    if voxel_grid.ndim == 4 and voxel_grid.shape[0] == 1:
        voxel_grid = voxel_grid.squeeze(0)

    # Threshold the voxel grid to create a binary grid
    binary_voxel_grid = (voxel_grid > 0.5).cpu().numpy().astype(np.int32)

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
        write_vtk_file(smoothed_polydata, f"smoothed_surface_mesh_epoch.vtp")
    except ValueError as e:
        print(f"Marching Cubes algorithm failed: {e}")


if __name__ == "__main__":
    device = torch.device( "cuda" if torch.cuda.is_available else "cpu")
    
    from_file = True
    
    if from_file:
        voxel_file = sys.argv[1]
        numpy_array = np.load(voxel_file) 
        #   print(numpy_array.shape)
        voxel_grid = torch.from_numpy(numpy_array).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        voxel_grid = torch.zeros(1, 1, 10, 10, 10)
        voxel_grid[:, :, 3:6, 3:6, 3:6] = 1.0
        voxel_grid[:, :, 4, 4, 4] = 0.0

    # write to npy file
    voxel_np = voxel_grid[0, 0].detach().cpu().numpy().astype(np.uint8)
    np.save(f"voxel.npy", voxel_np)

    # plot voxel.npy  
    # plot_voxel(voxel_np, title=f"Fake voxel")


    # write_to_vtk(voxel_grid)
    # print(voxel_grid)
    # resin_score = resintrap_score_3d(voxel_grid)
        

    # result = unified_resintrap_score(voxel_grid)

    # print("2D Score:", result["score_2d"].item())
    # print("3D Score:", result["score_3d"].item())
    # print("Combined Score:", result["combined_score"].item())

    # Convert for visualization (remove batch/channel and to numpy)
    # trap_mask_np = result["combined_mask"][0, 0].cpu().numpy().astype(np.uint8)

    if True:
        result = resin_trap_score(voxel_grid, resintrap_weight=100)
        print("resin trap score:", result["rtp"].item())
        print("enclosed_voids:", result["enclosed_voids"])


        # Convert for visualization (remove batch/channel and to numpy)
        trap_mask_np = result["enclosed_voids"][0, 0].cpu().numpy().astype(np.uint8)

        # Optionally save to .npy or VTK
        np.save("resin_trap_mask.npy", trap_mask_np)
        # plot_voxel_np(trap_mask_np, title="Detected Resin Traps")

        num_trap_voxels = int(np.count_nonzero(trap_mask_np))

        print(f"Total Resin Trap Voxels: {num_trap_voxels}")
        print(f"Resin Trap Penalty Score: {result['rtp']:.4f}")

        plot_enclosed_voids(trap_mask_np)
        
        #resin_score = resintrap_score(voxel_grid)
        # print("resin score: ", resin_score.item())

    # score, mask = refined_resintrap_2d(voxel_grid)
    # trap_mask_np = mask[0, 0].cpu().numpy().astype(np.uint8)

    # # Optionally save to .npy or VTK
    # # np.save("resin_trap_mask.npy", trap_mask_np)
    # # plot_voxel_np(trap_mask_np, title="Detected Resin Traps")

    # num_trap_voxels = int(np.count_nonzero(trap_mask_np))
    # print(f"Total Resin Trap Voxels: {num_trap_voxels}")
    # # print(f"Resin Trap Penalty Score: {result['rtp']:.4f}")

    # plot_enclosed_voids(trap_mask_np)
    