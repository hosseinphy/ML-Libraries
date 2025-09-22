"""Detects islands in 3D space within a binary voxel grid.

Uses 3D brute force search (bfs) without recusion.
"""

from collections import deque

import numpy as np

from . import utils

# Directions for all 26 neighbors in 3D space (6 face neighbors,
# 12 edge neighbors, 8 corner neighbors)
DIRECTIONS = [
    (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0),
    (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1),
    (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 1), (0, 1, -1),
    (0, 1, 0), (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0),
    (1, 1, 1)
]

def is_valid(x: int, y: int, z: int, grid: np.ndarray, visited: np.ndarray) -> bool:
    """Checks if point is in island."""
    size_z, size_y, size_x = grid.shape
    return 0 <= z < size_z    \
       and 0 <= y < size_y    \
       and 0 <= x < size_x    \
       and grid[z,y,x] == 1     \
       and not visited[z,y,x]

def bfs_3d(x: int, y: int, z: int, grid: np.ndarray, visited: np.ndarray) -> None:
    """Brute for search for all islands. Uses queue to avoid recusion.
    Visited grid to store all previously visited points.
    """
    start = (x,y,z)
    queue = deque([start])
    visited[z,y,x] = True
    
    points = [start]
    while queue:
        cx, cy, cz = queue.popleft()
        for dx, dy, dz in DIRECTIONS:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if is_valid(nx, ny, nz, grid, visited):
                visited[nz,ny,nx] = True
                queue.append((nx,ny,nz))
                points.append((nx,ny,nz))
    return points

def get_island_mask(grid: np.ndarray) -> np.ndarray:
    """Return boolean mask with same shape as input grid where True
    represents a 3D island.
    """
    
    size_z, size_y, size_x = grid.shape
    
    visited = np.zeros_like(grid, dtype=bool)
    island_mask = np.zeros_like(grid, dtype=bool)
    n_islands = 0
    
    # Use BFS on structure with frame, then count all other islands
    bfs_3d(0, 0, 0, grid, visited)
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                if grid[z,y,x] == 1 and not visited[z,y,x]:
                    island = bfs_3d(x, y, z, grid, visited)
                    n_islands += 1
                    for ix,iy,iz in island:
                        island_mask[iz,iy,ix] = True
                    
    # Main structure is always an island
    return island_mask, n_islands

def get_island_pts(grid: np.ndarray) -> np.ndarray:
    """Get all islands as Nx3 array of z,y,x points."""

    return np.transpose(np.nonzero(get_island_mask(grid)[0]))

def count_island_pts(grid: np.ndarray) -> int:
    """Get count of all islands voxels in grid."""

    return np.count_nonzero(get_island_mask(grid)[0])

if __name__ == "__main__":
    
    # Example 5x5 voxel map, 1 island
    grid = utils.add_frame(np.zeros((10,10,10)))
    grid[2,2,2] = 1
    grid[4,4,4] = 1
    grid[6,6,6] = 1
    grid[8,8,8] = 1

    print("Number of islands:", get_island_mask(grid)[1])
