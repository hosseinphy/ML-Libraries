import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


def diff_find_enclosed_voids(voxel_grid, seed_mask, iterations=10, threshold=0.5, steepness=20.):
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
        mask = torch.max(mask, activated)
        # Ensure thre fill does not go to the solid regions
        mask = mask * (1-voxel_grid)

    return mask


class Assessor(nn.Module):
    def __init__(self, input_shape=(32, 32, 32), support_kernel_size=3, target_occupancy=0.5, occupancy_weight=10., resintrap_weight=100.):
        super().__init__()
        self.input_shape = input_shape
        self.target_occupancy = target_occupancy
        self.support_kernel_size = support_kernel_size
        self.occupancy_weight = occupancy_weight
        self.resintrap_weight = resintrap_weight


    # def forward(self, voxel_grid, target_occupancy=None):
    #     if target_occupancy is None:
    #         target_occupancy = self.target_occupancy
    #     return {
    #         'occupancy': self.occupancy_score(voxel_grid, target_occupancy),
    #         'overhang': self.overhang_score(voxel_grid),
    #         'surface_score': self.surface_area_score(voxel_grid),
    #         'support_score': self.support_score_3d(voxel_grid)
    #     }

    # def forward(self, voxel_grid, target_occupancy=None):
    #     if target_occupancy is None:
    #         target_occupancy = self.target_occupancy
    #     return {
    #         'occupancy': self.occupancy_score(voxel_grid, target_occupancy),
    #         'overhang': self.overhang_score(voxel_grid),
    #         'surface_score': self.surface_area_score(voxel_grid),
    #         'support_score': self.support_score_3d(voxel_grid),
    #         # 'isolation_score': self.isolated_voxel_penalty(voxel_grid),
    #         'isolation_score': self.isolated_surface_voxel_penalty(voxel_grid)

    #     }

    def forward(self, voxel_grid, target_occupancy=None, occupancy_weight=None, resintrap_weight=None):
        if target_occupancy is None:
            target_occupancy = self.target_occupancy
        if occupancy_weight is None:
            occupancy_weight = self.occupancy_weight
        if resintrap_weight is None:
            resintrap_weight = self.resintrap_weight

        return {
            'occupancy': self.occupancy_score(voxel_grid, target_occupancy, occupancy_weight),
            'overhang': self.overhang_score(voxel_grid),
            'surface_score': self.surface_area_score(voxel_grid, resintrap_weight),
            'support_score': self.support_score_3d(voxel_grid),
            'isolation_score': self.isolated_surface_voxel_penalty(voxel_grid),
        }



    def get_kernel2d(self, device, flag=None):
        if flag == 1:
            return torch.ones(1, 1, self.support_kernel_size, self.support_kernel_size, device=device)
        elif flag == 0:
            return torch.zeros(1, 1, self.support_kernel_size, self.support_kernel_size, device=device)
        else:
            print("Problem with your option, chosse either 0 or 1 for flag")
            return None

    def get_kernel3d(self, device, flag=None):
        if flag == 1:
            return torch.ones(1, 1, self.support_kernel_size, self.support_kernel_size, self.support_kernel_size, device=device)
        elif flag == 0:
            return torch.zeros(1, 1, self.support_kernel_size, self.support_kernel_size, self.support_kernel_size, device=device)
        else:
            print("Problem with your option, chosse either 0 or 1 for flag")
            return None


    def surface_area_score(self, voxel_grid, resintrap_weight):

        """
        target_occupancy is the desired occupancy level (e.g., 0.25 for 25% solid material).
        """
        
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device


        seed_mask = torch.zeros_like(voxel_grid)
        seed_mask[:, :, 5, 5, 5] = 1
        # voxel_grid = voxel_grid.float()
        
        # Define a 3D kernel to count face-connected neighboring occupied voxels
        kernel = self.get_kernel3d(device,flag=0) #torch.zeros((1, 1, 3, 3, 3), device=voxel_grid.device)
        kernel[0, 0, 1, 1, 0] = 1  # Left
        kernel[0, 0, 1, 1, 2] = 1  # Right
        kernel[0, 0, 1, 0, 1] = 1  # Front
        kernel[0, 0, 1, 2, 1] = 1  # Back
        kernel[0, 0, 0, 1, 1] = 1  # Top
        kernel[0, 0, 2, 1, 1] = 1  # Bottom
        
        neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)
        
        # Each occupied voxel has up to 6 faces; subtract occupied neighbors to find exposed faces
        exposed_faces = 6 * voxel_grid - neighbor_count
       
        # Clamp negative values to zero (in case of over-subtraction)
        exposed_faces = torch.clamp(exposed_faces, min=0)

        # Sum the exposed faces and normalize by the number of occupied voxels
        surface_area = exposed_faces.sum() / voxel_grid.sum()
        
        
        # Compute enclosed voids penalty -> resin traps
        enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
        resin_trap_penalty = resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

        # Final score
        score = surface_area - resin_trap_penalty 

        return score
   
    # def occupancy_score(self, voxel_grid, target_occupancy):
    #     occupancy = voxel_grid.float().sum() / voxel_grid.numel()
    #     return self.occupancy_weight * (occupancy - target_occupancy) ** 2

    def occupancy_score(self, voxel_grid, target_occupancy, occupancy_weight):
        occupancy = voxel_grid.float().sum() / voxel_grid.numel()
        return occupancy_weight * (occupancy - target_occupancy) ** 2


    # def overhang_score(self, voxel_grid):
    #     B, C, D, H, W = voxel_grid.shape
    #     device = voxel_grid.device
    #     kernel2d = self.get_kernel2d(device, flag=1)
    #     print(kernel2d)
    #     padding = kernel2d.shape[-1] // 2
    #     penalty = 0.0

    #     for z in range(1, D):
    #         curr = voxel_grid[:, :, z, :, :]
    #         below = voxel_grid[:, :, z - 1, :, :]
    #         below_support = F.conv2d(below, kernel2d, padding=padding)
    #         support_ratio = below_support / kernel2d.sum()
    #         unsupported = curr * (1.0 - support_ratio)
    #         penalty += unsupported.sum()

    #     return penalty / (B * D * H * W)


    # Penalize overhangs based on support from below
    # Make the penalty more sgtrict by creating the threshold for support ratio
    # If the support ratio is below a certain threshold, the voxel is considered unsupported
    # and contributes to the penalty.
    # The threshold can be adjusted to make the penalty more or less strict. 
    def overhang_score(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device
        kernel2d = self.get_kernel2d(device, flag=1)
        padding = kernel2d.shape[-1] // 2
        penalty = 0.0

        threshold = 0.1 #0.3  # voxels with less than 30% support are penalized
        steepness = 50.0 #30.0
        margin = 0.05 # Mask Marginal Support Zones More Aggressively
        for z in range(1, D):   
            curr = voxel_grid[:, :, z, :, :]
            below = voxel_grid[:, :, z - 1, :, :]
            below_support = F.conv2d(below, kernel2d, padding=padding)
            support_ratio = below_support / kernel2d.sum()
            # unsupported = curr * (support_ratio < threshold).float()
            # unsupported = curr * torch.sigmoid(-steepness * (support_ratio - 0.3))
            unsupported = curr * torch.sigmoid(steepness * (threshold - support_ratio - margin))
            penalty += unsupported.sum()

        return penalty / (B * D * H * W)

    

    def support_score_3d(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device

        kernel = self.get_kernel3d(device, flag=0)
        kernel[0, 0, 1, 1, 0] = 1  # Left
        kernel[0, 0, 1, 1, 2] = 1  # Right
        kernel[0, 0, 1, 0, 1] = 1  # Front
        kernel[0, 0, 1, 2, 1] = 1  # Back
        kernel[0, 0, 0, 1, 1] = 1  # Top
        kernel[0, 0, 2, 1, 1] = 1  # Bottom

        # count neighbors for each voxel
        neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)

        # Voxels that are occupied but have low neighbor support
        unsupported = voxel_grid * (neighbor_count < 2)  # e.g., <2 neighbors is weak

        # Normalize by total number of voxels
        penalty = unsupported.sum() / voxel_grid.numel()

        return penalty

    def isolated_voxel_penalty(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device
        kernel = self.get_kernel3d(device, flag=0)

        # Use 6-connectivity
        kernel[0, 0, 1, 1, 0] = 1
        kernel[0, 0, 1, 1, 2] = 1
        kernel[0, 0, 1, 0, 1] = 1
        kernel[0, 0, 1, 2, 1] = 1
        kernel[0, 0, 0, 1, 1] = 1
        kernel[0, 0, 2, 1, 1] = 1

        neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)
        
        # Voxels with ≤1 neighbors (very isolated or dangling)
        occupied = voxel_grid.sum()

        isolated = voxel_grid * (neighbor_count <= 1)
        
        # before
        # penalty = isolated.sum() / voxel_grid.numel()
        
        # after
        penalty = (isolated.sum() / (occupied + 1e-8))  # avoid division by zero
        return penalty


    # def isolated_surface_voxel_penalty(self, voxel_grid):
    #     B, C, D, H, W = voxel_grid.shape
    #     device = voxel_grid.device

    #     # Step 1: Get 6-neighbor kernel
    #     kernel = self.get_kernel3d(device, flag=0)
    #     kernel[0, 0, 1, 1, 0] = 1  # left
    #     kernel[0, 0, 1, 1, 2] = 1  # right
    #     kernel[0, 0, 1, 0, 1] = 1  # front
    #     kernel[0, 0, 1, 2, 1] = 1  # back
    #     kernel[0, 0, 0, 1, 1] = 1  # top
    #     kernel[0, 0, 2, 1, 1] = 1  # bottom

    #     neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)

    #     # Step 2: Get isolated voxels (<= 1 neighbor)
    #     isolated = (voxel_grid == 1) & (neighbor_count <= 1)

    #     # Step 3: Compute surface mask
    #     full_kernel = self.get_kernel3d(device, flag=1)
    #     neighbor_count_all = F.conv3d(voxel_grid, full_kernel, padding=1)
    #     surface = (voxel_grid == 1) & (neighbor_count_all < full_kernel.sum())

    #     # Step 4: Dilate the surface mask to get nearby region
    #     dilated_surface = F.conv3d(surface.float(), full_kernel, padding=1) > 0

    #     # Step 5: Only penalize isolated voxels near surface
    #     penalized_voxels = isolated & dilated_surface
    #     penalty = penalized_voxels.sum().float() / (voxel_grid.sum() + 1e-8)

    #     return penalty

   
    def isolated_surface_voxel_penalty(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device

        kernel = self.get_kernel3d(device, flag=0)
        # 6-connectivity
        kernel[0, 0, 1, 1, 0] = 1
        kernel[0, 0, 1, 1, 2] = 1
        kernel[0, 0, 1, 0, 1] = 1
        kernel[0, 0, 1, 2, 1] = 1
        kernel[0, 0, 0, 1, 1] = 1
        kernel[0, 0, 2, 1, 1] = 1

        neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)
        occupied = voxel_grid > 0
        isolated = occupied & (neighbor_count <= 1)

        # Compute boundary mask
        dilated = F.conv3d(voxel_grid, kernel, padding=1) > 0
        boundary_mask = dilated & (~occupied)

        # Voxels adjacent to boundary
        boundary_zone = F.conv3d(boundary_mask.float(), kernel, padding=1) > 0

        # Avoid penalizing margin voxels
        core_mask = torch.zeros_like(voxel_grid, dtype=torch.bool)
        core_mask[:, :, 1:-1, 1:-1, 1:-1] = True

        boundary_isolated = isolated & boundary_zone & core_mask

        # Normalize by total occupied voxel count
        penalty = boundary_isolated.sum() / (occupied.sum() + 1e-8)
        return penalty
