import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

# adding a condition to max surface area that check for enclosed voids
def find_enclosed_voids(voxel_grid):
    voxel_np = (voxel_grid[0, 0].detach().cpu().numpy()).astype(np.uint8)
    D, H, W = voxel_np.shape
    visited = np.zeros((D, H, W), dtype=bool)
    directions= [(-1, 0, 0), (0, 0, 1), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    queue = deque()
    for z in range(D):
        for y in range(H):
            for x in range(W):
                is_boundary = z == 0 or z == D-1 or y == 0 or y == H -1 or x == 0 or x == W - 1
                if is_boundary and voxel_np[z, y, x] == 0 and not visited[z, y, x]:
                    queue.append((z, y, x))
                    visited[z, y, x] = True
                    while queue:
                        cz, cy, cx = queue.popleft()
                        for dz, dy, dx in directions:
                            nz, ny, nx = cz + dz, cy + dy, cx + dx
                            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                                if voxel_np[nz, ny, nx] == 0 and not visited[nz, ny, nx]:
                                    visited[nz, ny, nx] = True
                                    queue.append((nz, ny, nx))

    # Enclosed voids are unvisited empty voxels
    enclosed_voids = (voxel_np == 0) & (~visited)
    return enclosed_voids



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
    def __init__(self, input_shape=(32, 32, 32), support_kernel_size=3, target_occupancy=0.5, occupancy_weight=10.):
        super().__init__()
        self.input_shape = input_shape
        self.target_occupancy = target_occupancy
        self.support_kernel_size = support_kernel_size
        self.occupancy_weight = occupancy_weight

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

    def resintrap_score_2d(self, voxel_grid):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device
        kernel2d = self.get_kernel2d(device)
        padding = kernel2d.shape[-1] // 2
        penalty = 0.0

        for z in range(D):
            layer = voxel_grid[:, :, z, :, :]  # [B, 1, H, W]
            voids = 1.0 - layer
            local_sum = F.conv2d(layer, kernel2d, padding=padding)
            enclosed = (local_sum >= kernel2d.sum().item()).float()
            penalty += (voids * enclosed).sum()

        return penalty / (B * D * H * W)
    
    #ver 1.0 - hard traps with discrete function  -non-differentiable
    # def resintrap_score_3d(self, voxel_grid):
    #     B, C, D, H, W = voxel_grid.shape
    #     device = voxel_grid.device
    #     kernel3d = self.get_kernel3d(device)
    #     voids = 1.0 - voxel_grid
    #     neighbor_solid_count = F.conv3d(voxel_grid, kernel3d, padding=1)
    #     max_neighbors = kernel3d.sum().item()
    #     enclosed = (neighbor_solid_count >= max_neighbors - 1).float()  # tolerate center being 0
    #     resin_traps = voids * enclosed
    #     return resin_traps.sum() / (B * D * H * W)

    #ver 2.0  soft trap
    # def resintrap_score_3d(self, voxel_grid):
    #     B, C, D, H, W = voxel_grid.shape
    #     device = voxel_grid.device
    #     kernel = torch.ones(1, 1, 3, 3, 3, device=device)

    #     voids = 1.0 - voxel_grid  # soft voids
    #     neighbor_solid_count = F.conv3d(voxel_grid, kernel, padding=1)
    #     max_neighbors = kernel.sum().item()

    #     # Soft resin trap signal: how enclosed each void is
    #     enclosed_strength = neighbor_solid_count / max_neighbors
    #     soft_traps = voids * enclosed_strength  # keep gradient

    #     return soft_traps.sum() / (B * D * H * W)

    #ver 3.0 -  hard trap with differentiable function
    #def resintrap_score_3d(self, voxel_grid, threshold=24.0, steepness=30.0):
    def resintrap_score_3d(self, voxel_grid, threshold=20.0, steepness=20.0):
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device
        kernel = torch.ones(1, 1, 3, 3, 3, device=device)

        voids = 1.0 - voxel_grid
        neighbor_solid_count = F.conv3d(voxel_grid, kernel, padding=1)

        # print("Max solid neighbor count:", neighbor_solid_count.max().item())
        # print("Mean solid neighbor count:", neighbor_solid_count.mean().item())

        # Soft but sharp approximation of ">= 26"
        enclosed_soft = torch.sigmoid(steepness * (neighbor_solid_count - threshold))

        resin_traps = voids * enclosed_soft

        return resin_traps.sum() / (B * D * H * W)

    def unified_resintrap_score(self, voxel_grid, alpha_2d=0.5, alpha_3d=0.5):
        score_2d = self.resintrap_score_2d(voxel_grid)
        score_3d = self.resintrap_score_3d(voxel_grid)
        return alpha_2d * score_2d + alpha_3d * score_3d

    def occupancy_score(self, voxel_grid):
        occupancy = voxel_grid.sum() / voxel_grid.numel()
        #return torch.abs(occupancy - self.target_occupancy)
        return (occupancy - self.target_occupancy) ** 2
    
    # surface area ver 1.0
    # def surface_area_score (self, voxel_grid):
    #     B, C, D, H, W = voxel_grid.shape
        
    #     seed_mask = torch.zeros_like(voxel_grid)
    #     seed_mask[:, :, 5, 5, 5] = 1
        
    #     device = voxel_grid.device
    #     kernel_3d = torch.ones(1, 1, 3, 3, 3, device=device)
    #     max_neighbor_counts =kernel_3d.sum().item()
    #     neighbor_solid_count = F.conv3d(voxel_grid, kernel_3d, padding=1)
    #     surface_area = neighbor_solid_count/max_neighbor_counts  

    #     # enclosed_voids = find_enclosed_voids(voxel_grid)

    #     enclosed_voids = diff_fine_enclosed_voids(voxel_grid, seed_mask, iterations=20, steepness=10)
    #     # print(torch.sum(filled_mask).item())
    #     # print(enclosed_voids.sum())
    #     # Compute the final score by combining surface area and enclosed voids
    #     # Apply a large penalty if any resin traps are present
    #     penalty_factor = 10  # Adjust this factor as needed
    #     resin_trap_penalty = penalty_factor * enclosed_voids.sum() / (B * D * H * W)

    #     score = surface_area.sum() / (B * D * H * W) - resin_trap_penalty

    #     return score
       

    # surface area ver 2.0 
    def surface_area_score(self, voxel_grid):

        """
        
        target_occupancy is the desired occupancy level (e.g., 0.25 for 25% solid material).
        occupancy_weight controls the strength of the penalty for deviating from the target occupancy.
        diff_find_enclosed_voids is a differentiable function that identifies enclosed voids (resin traps).
        
        """
        
        B, C, D, H, W = voxel_grid.shape
        device = voxel_grid.device

        seed_mask = torch.zeros_like(voxel_grid)
        seed_mask[:, :, 5, 5, 5] = 1

        # Compute surface area
        kernel_3d = torch.ones(1, 1, 3, 3, 3, device=device)
        neighbor_solid_count = F.conv3d(voxel_grid, kernel_3d, padding=1)
        surface_area = neighbor_solid_count / kernel_3d.sum().item()

        # Compute occupancy
        occupancy = voxel_grid.sum() / (B * D * H * W)

        # Compute occupancy penalty
        occupancy_penalty = occupancy_weight * (occupancy - target_occupancy) ** 2

        # Compute enclosed voids penalty
        enclosed_voids = diff_find_enclosed_voids(voxel_grid, seed_mask)
        resin_trap_penalty = 100 * enclosed_voids.sum() / (B * D * H * W)

        # Final score
        score = surface_area.sum() / (B * D * H * W) - resin_trap_penalty - occupancy_penalty

        return score



   
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


