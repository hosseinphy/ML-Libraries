import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# def diff_find_enclosed_voids(voxel_grid, seed_mask, iterations=10, threshold=0.5, steepness=5.0):
#     """
#     Differentiable flood-fill that avoids leaking into solids by updating only void regions.
#     """
#     B, C, D, H, W = voxel_grid.shape
#     device = voxel_grid.device

#     # 6-connected kernel
#     kernel = torch.tensor([[[[0, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 0]],
#                             [[0, 1, 0],
#                              [1, 0, 1],
#                              [0, 1, 0]],
#                             [[0, 0, 0],
#                              [0, 1, 0],
#                              [0, 0, 0]]]], dtype=torch.float32, device=device).unsqueeze(0)

#     mask = seed_mask.clone()

#     for _ in range(iterations):
#         conv_result = F.conv3d(mask, kernel, padding=1)
#         activated = torch.sigmoid(steepness * (conv_result - threshold))
#         mask = torch.where(voxel_grid == 0, torch.max(mask, activated), mask)

#     return mask




def diff_find_enclosed_voids(voxel_grid, seed_mask, iterations=10, threshold=1.0, steepness=20.0):
    B, C, D,  H, W = voxel_grid.shape

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
         # Convolve the current mask
        conv_result = F.conv3d(mask, kernel, padding=1)
         # Apply sigmoid activation
        activated = torch.sigmoid(steepness * (conv_result - threshold) )
        
        # update mask

        # mask = torch.max(mask, activated)
        # # Ensure thre fill does not go to the solid regions
        # mask = mask * (1-voxel_grid)
    
        # The code block above may cause Leak to solid areas!!!!
        mask = torch.where(voxel_grid == 0, torch.max(mask, activated), mask)

    return mask



# def improved_seed_mask(voxel_grid):
#     """
#     Generate a seed mask for flood-fill initialized only at the 8 volume corners.
#     This minimizes over-activation from surface-adjacent voids.
#     """
#     B, C, D, H, W = voxel_grid.shape
#     seed_mask = torch.zeros_like(voxel_grid)

#     corners = [
#         (0, 0, 0),
#         (0, 0, W - 1),
#         (0, H - 1, 0),
#         (0, H - 1, W - 1),
#         (D - 1, 0, 0),
#         (D - 1, 0, W - 1),
#         (D - 1, H - 1, 0),
#         (D - 1, H - 1, W - 1),
#     ]

#     for d, h, w in corners:
#         seed_mask[:, :, d, h, w] = 1

#     return seed_mask


def resin_trap_score(voxel_grid, resintrap_weight=100):
    B, C, D, H, W = voxel_grid.shape
    seed_mask = torch.zeros_like(voxel_grid)
    seed_mask[:, : , 5, 5, 5] = 1
    # seed_mask[:, :, 0, :, :] = 1
    # seed_mask[:, :, -1, :, :] = 1
    # seed_mask[:, :, :, 0, :] = 1
    # seed_mask[:, :, :, -1, :] = 1
    # seed_mask[:, :, :, :, 0] = 1
    # seed_mask[:, :, :, :, -1] = 1
    
    # seed_mask = improved_seed_mask(voxel_grid)


    enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
    rtp = resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

    return {
        "rtp": rtp.item(),
        "enclosed_voids": enclosed_voids[0, 0].detach().cpu().numpy()
    }

def plot_enclosed_voids(trap_mask_np, title="Resin Traps"):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.nonzero(trap_mask_np)
    ax.scatter(x, y, z, c='red', marker='o', s=5, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load your voxel file
    voxel_np = np.load("../results_tune_config_baseline_occupancy_overhang_resintrap/fake_voxel_epoch_70000.npy")  # Replace with correct path if needed
    voxel_tensor = torch.from_numpy(voxel_np).float().unsqueeze(0).unsqueeze(0).to("cuda")

    result = resin_trap_score(voxel_tensor)
    trap_mask_np = result["enclosed_voids"]
    num_trap_voxels = int(np.count_nonzero(trap_mask_np))

    print(f"Total Resin Trap Voxels: {num_trap_voxels}")
    print(f"Resin Trap Penalty Score: {result['rtp']:.4f}")

    plot_enclosed_voids(trap_mask_np)
