import numpy as np


def add_frame(grid, width=1):
    """Add solid top and bottom layer to grid, as well on two vertial sides.

    Increases size of grid by 1 for each dimension.
    """

    grid = np.pad(grid, width, 'constant', constant_values=0)

    # Bottom and Top
    grid[0:width, :, :] = 1
    grid[-width::, :, :] = 1

    # Sides
    grid[:, :, 0:width] = 1
    grid[:, :,-width::] = 1

    return grid

def remove_frame(grid):
    return grid[1:-1, 1:-1, 1:-1]