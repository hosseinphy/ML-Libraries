"""Determines the number of resin traps in a 3D binary voxel map.

Resin traps are holes or hollow regions in a slice. If a hole exists
on a given slice and the same location directly above it, it counts
as two holes.
"""
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def get_resin_trap_mask(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Return boolean mask with same shape as input grid where True
    represents a resin trap.
    """

    grid = grid.astype(np.uint8)
    resin_trap_mask = np.zeros_like(grid)
    n_resin_traps = 0
    for z, slic in enumerate(grid):

        contours, hierarchy = cv2.findContours(slic,cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue
        
        for c,h in zip(contours, hierarchy[0]):
            if h[3] != -1:
                cv2.drawContours(resin_trap_mask[z], [c], 0, 1, -1)
                n_resin_traps += 1

    # draw_contours includes border of resin trap, use AND to get hole only
    resin_trap_mask = np.logical_and(resin_trap_mask, np.logical_not(grid))

    return resin_trap_mask, n_resin_traps




def get_resin_trap_pts(grid: np.ndarray) -> np.ndarray:
    """Get all resin traps as Nx3 array of z,y,x points."""

    return np.transpose(np.nonzero(get_resin_trap_mask(grid)[0]))

def count_resin_trap_pts(grid: np.ndarray) -> int:
    """Get count of all resin traps voxels in grid."""

    return np.count_nonzero(get_resin_trap_mask(grid[0]))


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

    # Example 10x10x10 grid, 1 holes
    # grid = np.zeros((10,10,10))

    # grid[3:6,3:6,3:6] = 1
    # grid[4,4,4] = 0

    voxel  = sys.argv[1] if len(sys.argv) > 1 else 'voxel.npy'
    grid = np.load(voxel)

    num_resin_traps = get_resin_trap_mask(grid)[1]
    resin_mask = get_resin_trap_mask(grid)[0] #get_resin_trap_pts(grid)
    # print(resin_mask.shape)
    
    print(f'Total number of resin traps: {num_resin_traps}')

    plot_voxel_with_resin_traps(grid, resin_mask)

    

    
