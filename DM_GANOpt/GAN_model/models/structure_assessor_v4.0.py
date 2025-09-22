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
    def __init__(self, input_shape=(32, 32, 32), support_kernel_size=3, target_occupancy=0.5, occupancy_weight=10., resintrap_weight=1000.):
        super().__init__()
        self.input_shape = input_shape
        self.target_occupancy = target_occupancy
        self.support_kernel_size = support_kernel_size
        self.occupancy_weight = occupancy_weight
        self.resintrap_weight = resintrap_weight

    def forward(self, voxel_grid):
        return {
            # 'occupancy': self.occupancy_score(voxel_grid),
            # 'resintrap': self.unified_resintrap_score(voxel_grid),
            'overhang': self.overhang_score(voxel_grid),
            'surface_score': self.surface_area_score(voxel_grid)
        }

    def get_kernel2d(self, device):
        return torch.ones(1, 1, self.support_kernel_size, self.support_kernel_size, device=device)

    def get_kernel3d(self, device):
        return torch.ones(1, 1, 3, 3, 3, device=device)


    def surface_area_score(self, voxel_grid):

        """
        target_occupancy is the desired occupancy level (e.g., 0.25 for 25% solid material).
        """
        
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device

        # voxel_grid = voxel_grid.float()
        
        # Define a 3D kernel to count face-connected neighboring occupied voxels
        kernel = torch.zeros((1, 1, 3, 3, 3), device=voxel_grid.device)
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
        
             # Compute occupancy
        occupancy = voxel_grid.float().sum() / voxel_grid.numel()

        # Compute occupancy penalty
        occupancy_penalty =  (occupancy - self.target_occupancy) ** 2

        # Compute enclosed voids penalty
        # enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
        # resin_trap_penalty = self.resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

        # Final score
        score = surface_area - self.occupancy_weight *  occupancy_penalty


        return score




    # # surface area ver 2.0 
    # def surface_area_score(self, voxel_grid):

    #     """
    #     target_occupancy is the desired occupancy level (e.g., 0.25 for 25% solid material).
    #     occupancy_weight controls the strength of the penalty for deviating from the target occupancy.
    #     diff_find_enclosed_voids is a differentiable function that identifies enclosed voids (resin traps).
    #     """
        
    #     B, C, D, H, W = voxel_grid.shape
    #     device = voxel_grid.device

    #     # Compute surface area
    #     kernel_3d = torch.ones(1, 1, 3, 3, 3, device=device)
    #     neighbor_solid_count = F.conv3d(voxel_grid, kernel_3d, padding=1)
        
    #     # new definition
    #     # Each occupied voxel has up to 6 faces; subtract occupied neighbors to find exposed faces
    #     exposed_faces = 6 * voxel_grid - neighbor_solid_count

    #     # Clamp negative values to zero (in case of over-subtraction)
    #     exposed_faces = torch.clamp(exposed_faces, min=0)

    #     # Sum the exposed faces and normalize by the number of occupied voxels
    #     surface_area = exposed_faces.sum() / voxel_grid.sum()

        
    #     # Compute occupancy
    #     occupancy = voxel_grid.float().sum() / voxel_grid.numel()

    #     # Compute occupancy penalty
    #     occupancy_penalty = self.occupancy_weight * (occupancy - self.target_occupancy) ** 2

    #     # Compute enclosed voids penalty
    #     # enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
    #     # resin_trap_penalty = self.resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

    #     # Final score
    #     score = surface_area - occupancy_penalty

    #     return score



   
    def overhang_score(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device
        kernel2d = self.get_kernel2d(device)
        padding = kernel2d.shape[-1] // 2
        penalty = 0.0

        for z in range(1, D):
            curr = voxel_grid[:, :, z, :, :]
            below = voxel_grid[:, :, z - 1, :, :]
            below_support = F.conv2d(below, kernel2d, padding=padding)
            support_ratio = below_support / kernel2d.sum()
            unsupported = curr * (1.0 - support_ratio)
            penalty += unsupported.sum()

        return penalty / (B * D * H * W)



# import torch.nn as nn
# import torch
# import torch.nn.functional as F


# class Assessor(nn.Module):
#     def __init__(self, input_shape=(32, 32, 32), support_kernel_size=3, target_occupancy=0.1):
#         super().__init__()
#         self.input_shape = input_shape
#         self.target_occupancy = target_occupancy
#         self.kernel = torch.ones(1, 1, support_kernel_size, support_kernel_size)

#     def forward(self, voxel_grid):
#         return {
#             'occupancy': self.occupancy_score(voxel_grid),
#             'resintrap': self.unified_resintrap_score(voxel_grid),
#             'overhang': self.overhang_score(voxel_grid),
#         }

#     # def resintrap_score(self, voxel_grid):
#     #     penalty = 0.0
#     #     B, C, D, H, W = voxel_grid.shape
#     #     device = voxel_grid.device
#     #     kernel = self.kernel.to(device)
#     #     padding = kernel.shape[-1] // 2
#     #     print("padding for resintrap: ", padding)
#     #     for z in range(D):
#     #         layer = voxel_grid[:, :, z, :, :]
#     #         inverted = 1.0 - layer
#     #         local_sum = F.conv2d(layer, kernel, padding=padding)
#     #         fully_enclosed = (local_sum == kernel.sum()).float()
#     #         penalty += (inverted * fully_enclosed).sum()

#     #     return penalty / (B * D * H * W)
    

#     def resintrap_score_2d(self, voxel_grid):
#         B, C, D, H, W = voxel_grid.shape
#         device = voxel_grid.device
#         kernel2d = torch.ones(1, 1, 3, 3, device=device)
#         padding = kernel2d.shape[-1] // 2
#         penalty = 0.0
#         trap_mask = torch.zeros_like(voxel_grid)

#         for z in range(D):
#             layer = voxel_grid[:, :, z, :, :]         # [B, 1, H, W]
#             voids = 1.0 - layer
#             local_sum = F.conv2d(layer, kernel2d, padding=padding)
#             enclosed = (local_sum >= 8).float()       # Fully enclosed in 2D
#             traps = voids * enclosed                  # Only apply to voids
#             trap_mask[:, :, z, :, :] = traps
#             penalty += traps.sum()

#         score = penalty / (B * D * H * W)
#         return score#, trap_mask



#     def resintrap_score_3d(self, voxel_grid):
#         """
#         Detects 3D hard resin traps: voids (zeros) that are completely enclosed by solid (ones) in 3D.
#         Args:
#             voxel_grid: torch.Tensor of shape [B, 1, D, H, W], values in [0, 1]
#         Returns:
#             Scalar differentiable resin trap score (higher = worse)
#         """
#         B, C, D, H, W = voxel_grid.shape
#         device = voxel_grid.device

#         # Define a 3D kernel to check 3x3x3 neighborhood (excluding center if desired)
#         kernel = torch.ones((1, 1, 3, 3, 3), device=device)
#         # kernel[0, 0, 1, 1, 1] = 0  # Optionally exclude center voxel

#         # Identify candidate voids
#         voids = 1.0 - voxel_grid  # 1 = void, 0 = solid

#         # Count how many of a void's neighbors are solid
#         neighbor_solid_count = F.conv3d(voxel_grid, kernel, padding=1)

#         # Number of neighbors (26 if we exclude center, 27 if not)
#         max_neighbors = kernel.sum() -1

#         # Fully enclosed void: it's a void and surrounded by solids 
#         # (greater sign is to account for to allow for some numerical tolerance or flexibility)
#         enclosed_mask = (neighbor_solid_count >= max_neighbors).float()

#         resin_traps = voids * enclosed_mask
#         score = resin_traps.sum() / (B * D * H * W)
#         # Normalize and return scalar score
#         return score#, resin_traps


#     def unified_resintrap_score(self, voxel_grid, alpha_2d=0.5, alpha_3d=0.5):
#         # score_2d, mask_2d = self.resintrap_score_2d(voxel_grid)
#         # score_3d, mask_3d = self.resintrap_score_3d(voxel_grid)
#         score_2d = self.resintrap_score_2d(voxel_grid)
#         score_3d = self.resintrap_score_3d(voxel_grid)
        
#         combined_score = alpha_2d * score_2d + alpha_3d * score_3d
#         # combined_mask = ((alpha_2d * mask_2d + alpha_3d * mask_3d) > 0).float()

#         # return {
#             # "score_2d": score_2d,
#             # "score_3d": score_3d,
#             # "combined_score": combined_score,
#             # "mask_2d": mask_2d,
#             # "mask_3d": mask_3d,
#             # "combined_mask": combined_mask
#         # }
#         return combined_score

#     def occupancy_score(self, voxel_grid):
#         occupancy = voxel_grid.float().sum() / voxel_grid.numel()
#         return torch.abs(occupancy - self.target_occupancy)

#     def overhang_score(self, voxel_grid):
#         penalty = 0.0
#         B, C, D, H, W = voxel_grid.shape
#         device = voxel_grid.device
#         kernel = self.kernel.to(device)
#         padding = kernel.shape[-1] // 2

#         for z in range(1, D):
#             curr = voxel_grid[:, :, z, :, :]
#             below = voxel_grid[:, :, z - 1, :, :]
#             below_support = F.conv2d(below, kernel, padding=padding)
#             support_ratio = below_support / kernel.sum()
#             unsupported = curr * (1.0 - support_ratio)
#             penalty += unsupported.sum()

#         return penalty / (B * D * H * W)



# # class Assessor(nn.Module):
# #     def __init__(self, input_shape=(32, 32, 32), support_kernel_size = 3, target_occupancy=0.1):
# #         super().__init__()

# #         self.input_shape =input_shape
# #         self.target_occupancy = target_occupancy
# #         self.kernel =torch.ones(1, 1, support_kernel_size, support_kernel_size)

    
# #     def forward(self, voxel_grid):
# #         return {
# #             'occupancy': self.occupancy_score(voxel_grid),
# #             'resintrap': self.resintrap_score(voxel_grid),
# #             'overhang':self.overhang_score(voxel_grid),
# #         }
    

# #     def resintrap_score(self, voxel_grid):
# #         penalty = 0.0
# #         B, C, D, H, W  = voxel_grid.shape
# #         device = voxel_grid.device
# #         kernel = self.kernel.to(device)

# #         for z in range(D):
# #             layer = voxel_grid[:, :, z, :, :]
# #             inverted = 1.0 - layer # 0-> solid, 1-> empty

# #             # Convolve to find empty pixels surrounded by solids
# #             local_sum = F.conv2d(layer, kernel, padding=1)
            
# #             # Fully surrounded if sum of neighbors = max (i.e., kernel.sum())
# #             fully_enclosed_score = inverted * (local_sum == kernel.sum()).float()

# #             penalty += fully_enclosed_score.sum()

# #         return penalty / (B * D * H * W)  # normalized

    

# #     def occupancy_score(self, voxel_grid):
# #         total_voxel = voxel_grid.numel()
# #         #occupied_voxel = (voxel_grid > 0.5).float().sum()
# #         occupied_voxel = voxel_grid.float().sum()
# #         occupancy = (occupied_voxel/total_voxel)
# #         return torch.abs(occupancy - self.target_occupancy)


# #     def overhang_score(self, voxel_grid):
         
# #         B, C, D, H, W = voxel_grid.shape
# #         penalty = 0.0
# #         device = voxel_grid.device
# #         kernel = self.kernel.to(device)

# #         for z in range(1, D):
# #             curr = voxel_grid[:, :, z, :, :]      # [B, 1, H, W]
# #             below = voxel_grid[:, :, z-1, :, :]   # [B, 1, H, W]

# #             # Calculate soft support from below via 2D convolution
# #             below_support = F.conv2d(below, kernel, padding=self.kernel.shape[-1]//2)
# #             # Normalize to [0, 1] — optional
# #             max_support = self.kernel.sum()
# #             support_ratio = below_support / max_support

# #             # Voxels in current layer with low support beneath
# #             unsupported_mask = curr * (1.0 - support_ratio)

# #             # Accumulate overhang penalty
# #             penalty += unsupported_mask.sum()

# #         return penalty / (B * D * H * W)  # normalize by volume

# if __name__== "__main__":
    
#     rt_voxel = torch.ones(1, 1, 32, 32, 32)
#     rt_voxel[:, :, 4:10, 4:10, 4:10] = 0
#     assessor = Assessor()
#     scores = assessor(rt_voxel)
#     print(scores["surface_score"].item())


    # # Create a voxel grid with a hollow cube
    # voxel_grid = torch.ones(1, 1, 32, 32, 32)
    # voxel_grid[:, :, 4:10, 4:10, 4:10] = 0
    # # voxel_grid[:, :, 10:22, 10:22, 10:22] = 1  # Solid cube inside

    # # Define seed mask (e.g., a point inside the hollow region)
    # seed_mask = torch.zeros_like(voxel_grid)
    # seed_mask[:, :, 5, 5, 5] = 1

    # # Perform differentiable flood fill
    # filled_mask = diff_fine_enclosed_voids(voxel_grid, seed_mask, iterations=20, steepness=10)
    # # print(torch.sum(filled_mask).item())
    # # print(filled_mask.sum().item())


