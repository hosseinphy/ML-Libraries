import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# # def diff_find_enclosed_voids(voxel_grid, seed_mask, iterations=10, threshold=0.5, steepness=20.):
# #     B, C, D,  H, W = voxel_grid.shape

# #     device = voxel_grid.device

# #     # Define a 3D kernel for 6-connectivity
# #     kernel = torch.tensor([[[[0, 0, 0],
# #                              [0, 1, 0],
# #                              [0, 0, 0]],
# #                             [[0, 1, 0],
# #                              [1, 0, 1],
# #                              [0, 1, 0]],
# #                             [[0, 0, 0],
# #                              [0, 1, 0],
# #                              [0, 0, 0]]]], dtype=torch.float32, device=device)
# #     kernel = kernel.unsqueeze(0)

# #     mask = seed_mask.clone()
# #     for _ in range(iterations):
# #          # Convolve the current mask
# #         conv_result = F.conv3d(mask, kernel, padding=1)
# #          # Apply sigmoid activation
# #         activated = torch.sigmoid(steepness * (conv_result - threshold) )
# #         # update mask
# #         mask = torch.max(mask, activated)
# #         # Ensure thre fill does not go to the solid regions
# #         mask = mask * (1-voxel_grid)

# #     return mask


# class Assessor(nn.Module):
#     def __init__(self, input_shape=(32, 32, 32), support_kernel_size=3, target_occupancy=0.5, occupancy_weight=10., resintrap_weight=1000.):
#         super().__init__()
#         self.input_shape = input_shape
#         self.target_occupancy = target_occupancy
#         self.support_kernel_size = support_kernel_size
#         self.occupancy_weight = occupancy_weight
#         self.resintrap_weight = resintrap_weight

#     def forward(self, voxel_grid):
#         return {
#             # 'occupancy': self.occupancy_score(voxel_grid),
#             # 'resintrap': self.unified_resintrap_score(voxel_grid),
#             'overhang': self.overhang_score(voxel_grid),
#             'surface_score': self.surface_area_score(voxel_grid)
#         }

#     def get_kernel2d(self, device):
#         return torch.ones(1, 1, self.support_kernel_size, self.support_kernel_size, device=device)

#     def get_kernel3d(self, device):
#         return torch.ones(1, 1, 3, 3, 3, device=device)


#     # surface area ver 2.0 
#     def surface_area_score(self, voxel_grid):

#         """
#         target_occupancy is the desired occupancy level (e.g., 0.25 for 25% solid material).
#         occupancy_weight controls the strength of the penalty for deviating from the target occupancy.
#         diff_find_enclosed_voids is a differentiable function that identifies enclosed voids (resin traps).
#         """
        
#         B, C, D, H, W = voxel_grid.shape
#         device = voxel_grid.device

#         # voxel_grid = voxel_grid.float()
        
#         # Define a 3D kernel to count face-connected neighboring occupied voxels
#         kernel = torch.zeros((1, 1, 3, 3, 3), device=voxel_grid.device)
#         kernel[0, 0, 1, 1, 0] = 1  # Left
#         kernel[0, 0, 1, 1, 2] = 1  # Right
#         kernel[0, 0, 1, 0, 1] = 1  # Front
#         kernel[0, 0, 1, 2, 1] = 1  # Back
#         kernel[0, 0, 0, 1, 1] = 1  # Top
#         kernel[0, 0, 2, 1, 1] = 1  # Bottom
        
#         neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)
        
#         # Each occupied voxel has up to 6 faces; subtract occupied neighbors to find exposed faces
#         exposed_faces = 6 * voxel_grid - neighbor_count

       
#         # Clamp negative values to zero (in case of over-subtraction)
#         exposed_faces = torch.clamp(exposed_faces, min=0)

#         # Sum the exposed faces and normalize by the number of occupied voxels
#         surface_area = exposed_faces.sum() / voxel_grid.sum()
        
#         # Compute occupancy
#         occupancy = voxel_grid.float().sum() / voxel_grid.numel()

#         # Compute occupancy penalty
#         # occupancy_penalty = self.occupancy_weight * (occupancy - self.target_occupancy) ** 2

#         # Compute enclosed voids penalty
#         # enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
#         # resin_trap_penalty = self.resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

#         # Final score
#         score = surface_area #- occupancy_penalty

#         return score

#         # Calculate the surface area of the occupied cube
  



   
#     def overhang_score(self, voxel_grid):
#         B, C, D, H, W = voxel_grid.shape
#         device = voxel_grid.device
#         kernel2d = self.get_kernel2d(device)
#         padding = kernel2d.shape[-1] // 2
#         penalty = 0.0

#         for z in range(1, D):
#             curr = voxel_grid[:, :, z, :, :]
#             below = voxel_grid[:, :, z - 1, :, :]
#             below_support = F.conv2d(below, kernel2d, padding=padding)
#             support_ratio = below_support / kernel2d.sum()
#             unsupported = curr * (1.0 - support_ratio)
#             penalty += unsupported.sum()

#         return penalty / (B * D * H * W)



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

    def forward(self, voxel_grid, target_occupancy=None):
        if target_occupancy is None:
            target_occupancy = self.target_occupancy
        return {
            'occupancy': self.occupancy_score(voxel_grid, target_occupancy),
            'overhang': self.overhang_score(voxel_grid),
            'surface_score': self.surface_area_score(voxel_grid),
            'support_score': self.support_score_3d(voxel_grid),
            'isolation_score': self.isolated_voxel_penalty(voxel_grid),
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


    def surface_area_score(self, voxel_grid):

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
        resin_trap_penalty = self.resintrap_weight * enclosed_voids.sum() / (B * D * H * W)

        # Final score
        score = surface_area - resin_trap_penalty 

        return score

   
    def occupancy_score(self, voxel_grid, target_occupancy):
        occupancy = voxel_grid.float().sum() / voxel_grid.numel()
        return self.occupancy_weight * (occupancy - target_occupancy) ** 2
    

    def overhang_score(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device
        kernel2d = self.get_kernel2d(device, flag=1)
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
        isolated = voxel_grid * (neighbor_count <= 1)
        penalty = isolated.sum() / voxel_grid.numel()
        return penalty





if __name__== "__main__":
        
    #test surface score
    if (0):
        # Initialize a 32x32x32 voxel grid filled with zeros (empty space)
        voxel_grid = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)

        # Define a smaller cube within the grid to be occupied
        voxel_grid[:, :, 8:24, 8:24, 8:24] = 1.0


    # test isolated voxel penalty
    # Create an empty voxel grid: [1, 1, 32, 32, 32]
    voxel_grid = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)

    # Add isolated voxels (completely disconnected)
    voxel_grid[0, 0, 5, 5, 5] = 1.0
    voxel_grid[0, 0, 10, 20, 10] = 1.0
    voxel_grid[0, 0, 25, 25, 25] = 1.0
    voxel_grid[0, 0, 15, 15, 30] = 1.0

    # Add a small connected cube (2×2×2)
    voxel_grid[0, 0, 12:14, 12:14, 12:14] = 1.0




    assessor = Assessor()
    scores = assessor(voxel_grid)
    
    # print(f"Surface Area: {surface_area}")
    # print(scores["surface_score"].item())
    
    print(scores["isolation_score"].item())





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

    # Extract the coordinates of occupied voxels
    occupied = voxel_grid[0, 0].nonzero()
    x, y, z = occupied[:, 0], occupied[:, 1], occupied[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
