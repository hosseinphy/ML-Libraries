import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from scipy.ndimage import label


# def diff_find_enclosed_voids(voxel_grid, seed_mask, iterations=25, threshold=0.5, steepness=20.):
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

    # # Apply hard threshold to avoid partial gradients causing leaks
    # binary_mask = (mask > 0.95).float()

    # return binary_mask
    return mask

def filter_small_clusters(mask_np, min_voxels=5):
    labeled, n = label(mask_np)
    for i in range(1, n + 1):
        if np.sum(labeled == i) < min_voxels:
            mask_np[labeled == i] = 0
    return mask_np



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
            'resintrap': self.resin_trap_score(voxel_grid, resintrap_weight)
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


    # old versin -- where I substract resintrap from surface score
    # def surface_area_score(self, voxel_grid, resintrap_weight):

    #     """
    #     target_occupancy is the desired occupancy level (e.g., 0.25 for 25% solid material).
    #     """
        
    #     B, C, D, H, W = voxel_grid.shape
    #     device = voxel_grid.device


    #     seed_mask = torch.zeros_like(voxel_grid)
    #     seed_mask[:, :, 5, 5, 5] = 1
    #     # voxel_grid = voxel_grid.float()
        
    #     # Define a 3D kernel to count face-connected neighboring occupied voxels
    #     kernel = self.get_kernel3d(device,flag=0) #torch.zeros((1, 1, 3, 3, 3), device=voxel_grid.device)
    #     kernel[0, 0, 1, 1, 0] = 1  # Left
    #     kernel[0, 0, 1, 1, 2] = 1  # Right
    #     kernel[0, 0, 1, 0, 1] = 1  # Front
    #     kernel[0, 0, 1, 2, 1] = 1  # Back
    #     kernel[0, 0, 0, 1, 1] = 1  # Top
    #     kernel[0, 0, 2, 1, 1] = 1  # Bottom
        
    #     neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)
        
    #     # Each occupied voxel has up to 6 faces; subtract occupied neighbors to find exposed faces
    #     exposed_faces = 6 * voxel_grid - neighbor_count
       
    #     # Clamp negative values to zero (in case of over-subtraction)
    #     exposed_faces = torch.clamp(exposed_faces, min=0)

    #     # Sum the exposed faces and normalize by the number of occupied voxels
    #     surface_area = exposed_faces.sum() / voxel_grid.sum()
        
        
    #     # Compute enclosed voids penalty -> resin traps
    #     enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
    #     resin_trap_penalty = resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

    #     # Final score
    #     score = surface_area - resin_trap_penalty 

    #     return score
   
    

    # New versin 
    def surface_area_score(self, voxel_grid, resintrap_weight):

        """
        target_occupancy is the desired occupancy level (e.g., 0.25 for 25% solid material).
        """
        
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device
        
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
      
        # Final score
        score = surface_area  

        return score
   
    

    def occupancy_score(self, voxel_grid, target_occupancy, occupancy_weight):
        occupancy = voxel_grid.float().sum() / voxel_grid.numel()
        return occupancy_weight * (occupancy - target_occupancy) ** 2
    

    def resin_trap_score(self, voxel_grid, resintrap_weight):
        """
        Penalize resin trap formation in the voxel grid.
        """
    
        B, C, D, H, W = voxel_grid.shape
        
        seed_mask = torch.zeros_like(voxel_grid)
        # Initiating flood-fill from a single voxel at (5,5,5) in every batch.
        # If the outer voids are not connected to this point, entire regions may be 
        # misclassified as resin traps, especially in larger grids or offset shapes.
        #--------------------------------------------------------------------------#
        seed_mask[:, :, 5, 5, 5] = 1
        # seed_mask[:, :, 0, :, :] = 1
        # seed_mask[:, :, -1, :, :] = 1
        # seed_mask[:, :, :, 0, :] = 1
        # seed_mask[:, :, :, -1, :] = 1
        # seed_mask[:, :, :, :, 0] = 1
        # seed_mask[:, :, :, :, -1] = 1


        # Compute enclosed voids penalty -> resin traps
        
        enclosed_voids_mask = diff_find_enclosed_voids(voxel_grid, seed_mask)

        # Remove tiny artifacts in mask (postprocessing for viz)
        # mask_np = enclosed_voids_mask[0, 0].detach().cpu().numpy()
        # mask_np_filtered = filter_small_clusters(mask_np.copy(), min_voxels=min_cluster_size)
        # filtered_mask = torch.from_numpy(mask_np_filtered).to(voxel_grid.device).unsqueeze(0).unsqueeze(0)


        # to make it differentiable -- see below
        # soft_mask = torch.sigmoid((enclosed_voids_mask - 0.5) * 20)
        # filtered_mask = soft_mask  # still differentiable


        # Avoid hard binarization in enclosed_voids_mask (above code block); keep it soft from diff_find_enclosed_voids.
        # soft_mask = enclosed_voids_mask
        # score = resintrap_weight * soft_mask.sum() / (B * D * H * W)

        # score = resintrap_weight * enclosed_voids_mask.mean()

        mean_void = enclosed_voids_mask.mean()
        score = resintrap_weight * torch.exp(10 * mean_void)  # Strong penalty if mean > 0.03
        # score = resintrap_weight * filtered_mask.sum() / (B * D * H * W)


        return enclosed_voids_mask, score

    
    # The version close to the structure builder
    def overhang_score(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device

        kernel2d = self.get_kernel2d(device, flag=1)  # 3×3 support check
        padding = kernel2d.shape[-1] // 2
        penalty = 0.0

        threshold = 0.11  # This still works if >0% support is needed

        for z in range(1, D):
            curr = voxel_grid[:, :, z, :, :]  # Current layer
            if curr.sum() == 0:
                continue

            below = voxel_grid[:, :, z - 1, :, :]  # Layer below
            # Convolve with 3x3 kernel to get support map
            below_support = F.conv2d(below, kernel2d, padding=padding)
            # Normalize: gives ratio of supported neighbors
            support_ratio = below_support / kernel2d.sum().item()

            # This mask marks voxels that are not supported even diagonally
            unsupported_mask = (support_ratio < threshold).float()
            unsupported = curr * unsupported_mask
            penalty += unsupported.sum()

        occupied = voxel_grid.sum()
        return penalty / (occupied + 1e-5)



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
