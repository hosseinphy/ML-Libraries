
import cv2

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import Slider

# 6-connectivity directions (no corners/edges)
DIRECTIONS = [
    (0, 0, -1), (0, 0, 1),
    (0, -1, 0), (0, 1, 0),
    (-1, 0, 0), (1, 0, 0),
]

# Directions for all 26 neighbors in 3D space (6 face neighbors,
# 12 edge neighbors, 8 corner neighbors)
# DIRECTIONS = [
#     (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0),
#     (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1),
#     (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 1), (0, 1, -1),
#     (0, 1, 0), (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1),
#     (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0),
#     (1, 1, 1)
# ]

def get_resin_trap_mask(grid: np.ndarray) -> np.ndarray:
    """
    Detects resin traps (internal voids) in a solid structure.
    Input: grid where 1 = solid, 0 = void
    Output: same-shape binary mask with 1s where resin traps exist
    """
    # Step 1: Invert the grid: 1 = void, 0 = solid
    voids = 1 - grid  

    # Step 2: Pad with external voids (1s) to define the "outside"
    padded = np.pad(voids, pad_width=1, mode='constant', constant_values=1)
    visited = np.zeros_like(padded, dtype=bool)

    # Step 3: Flood-fill from all exterior voids
    size_z, size_y, size_x = padded.shape

    def is_valid(x, y, z):
        return (
            0 <= z < size_z and 0 <= y < size_y and 0 <= x < size_x
            and padded[z, y, x] == 1 and not visited[z, y, x]
        )

    def bfs(start):
        queue = deque([start])
        visited[start] = True
        while queue:
            cx, cy, cz = queue.popleft()
            for dx, dy, dz in DIRECTIONS:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if is_valid(nx, ny, nz):
                    visited[nz, ny, nx] = True
                    queue.append((nx, ny, nz))

    # Flood-fill from all border positions
    for z in [0, size_z - 1]:
        for y in range(size_y):
            for x in range(size_x):
                if is_valid(x, y, z):
                    bfs((x, y, z))
    for z in range(size_z):
        for y in [0, size_y - 1]:
            for x in range(size_x):
                if is_valid(x, y, z):
                    bfs((x, y, z))
    for z in range(size_z):
        for y in range(size_y):
            for x in [0, size_x - 1]:
                if is_valid(x, y, z):
                    bfs((x, y, z))

    # Step 4: Enclosed voids = not visited
    resin_traps_padded = (padded == 1) & (~visited)

    # Remove padding
    resin_traps = resin_traps_padded[1:-1, 1:-1, 1:-1]
    n_resin_traps = np.sum(resin_traps)

    return resin_traps.astype(np.uint8), n_resin_traps



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


def plot_voxel_with_resin_traps(voxel_np, trap_mask_np, title="Voxel with Resin Traps"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Solid structure in light gray
    x_s, y_s, z_s = np.nonzero(voxel_np)
    ax.scatter(x_s, y_s, z_s, c='lightgray', alpha=0.3, s=10, label="Solid")

    # Resin traps in red
    x_r, y_r, z_r = np.nonzero(trap_mask_np)
    ax.scatter(x_r, y_r, z_r, c='red', alpha=0.9, s=20, label="Resin Traps")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.view_init(30, 45)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # import utils

    # 10×10×10 solid cube with internal voids
    # voxel = np.ones((10, 10, 10), dtype=np.uint8)
    # voxel[2, 2, 2] = 0
    # voxel[5, 5, 5] = 0
    # voxel[1, 1, 1] = 0  # edge void, should NOT be marked


    voxel_file = sys.argv[1] if len(sys.argv) > 1 else "voxel.npy"
    voxel = np.load(voxel_file)

    resin_mask,_ = get_resin_trap_mask(voxel)
    
    # Count trap voxels (with a soft threshold)
    num_trap_voxels = np.sum(resin_mask)
    print("Number of resin trap voxels:", num_trap_voxels)

    print("Number of resin trap voxels:", np.sum(resin_mask))
    plot_voxel_with_resin_traps(voxel, resin_mask)