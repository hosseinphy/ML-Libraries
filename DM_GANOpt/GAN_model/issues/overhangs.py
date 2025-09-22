"""Detects unsupported regions in a cube shaped voxel map.

Detects all contours in a slice (2D XY plane) of the voxel map and
checks for any  supporting pixels underneath to determine overhangs.
"""

from collections import deque

import numpy as np
import sys

BELOW_SUPPORT_VALUE = 1
EDGE_SUPPORT_VALUE = 1
CORNER_SUPPORT_VALUE = 1

REQUIRED_SUPPORT_FACTOR = 1

# ASPECT_RATIO_CUTOFF_1D = 4
# REQUIRED_SUPPORT_FACTOR_1D = 1
# REQUIRED_SUPPORT_FACTOR_2D = 1

MIN_AREA_TO_CONSIDER = 0

# Directions for the 8-connected neighbors in 2D
DIRECTIONS = [
    ( 0, -1), ( 0,  1), (-1,  0), ( 1,  0),
    (-1, -1), (-1,  1), ( 1, -1), ( 1,  1)
]

def is_valid(x: int, y: int, slic: np.ndarray) -> bool:
    """Checks if point is in island."""
    size_y, size_x = slic.shape
    return 0 <= y < size_y      \
       and 0 <= x < size_x      \
       and slic[y,x] == 1       \

def bfs_2d(x: int, y: int, slic: np.ndarray, visited: np.ndarray) -> list[(int,int)]:
    """Breadth-first search in 2D for all islands in a slice of the voxel
    map.
    """

    start = (x,y)
    queue = deque([start])
    visited[y,x] = True
    
    islands = [start]
    while queue:
        cx, cy = queue.popleft()
        for dy, dx in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if is_valid(nx, ny, slic) and not visited[ny,nx]: 
                visited[ny,nx] = True
                queue.append((nx,ny))
                islands.append((nx,ny))
    return islands

def bounding_box_dims(points):
    """Returns height and width of minimum bounding box of
    a set of points.
    """
    
    x_coordinates, y_coordinates = zip(*points)
    
    x_start = min(x_coordinates)
    x_end = max(x_coordinates)
    
    y_start = min(y_coordinates)
    y_end = max(y_coordinates)

    w = x_end - x_start + 1
    h = y_end - y_start + 1

    return w,h

def required_support_score(overhang: list[(int,int)]) -> float:
    """Returns minimum number of supports for a overhang to be supported.
    
    Since a given overhang pixel may be supported by not only a pixel
    below it, but possibly 8 other surrounding pixels below it as well
    supports may be counted multiple times in an island. This is accounted
    for in the REQUIRED_SUPPORT_FACTOR constant.

    The aspect ratio of the island (defined by a bounding box containing all
    island points) is normalized by how fully the overhang fills the bounding
    box (the extent). The more square the island, the less supports are required.
    The more line-like the island, the more the supports.
    """

    w,h = bounding_box_dims(overhang)
    rect_aspect_ratio = max(w,h) / min(w,h)

    area = len(overhang)
    rect_area = w * h
    extent = area / rect_area

    # Aspect ratio where overhang is considered line-like (1D)
    if rect_aspect_ratio / extent > ASPECT_RATIO_CUTOFF_1D:
        return area * REQUIRED_SUPPORT_FACTOR_1D

    return area * REQUIRED_SUPPORT_FACTOR_2D

def is_supported(overhang: list[(int,int)], prev_slic: np.ndarray) -> bool:
    """Determines if overhang has a sufficient number of supports.
    
    Small overhangs only require a single supporting pixel, where
    larger overhangs require more."""

    support_score = 0

    for (x, y) in overhang:

        # Check if supported directly below
        if prev_slic[y, x] != 0:
            support_score += BELOW_SUPPORT_VALUE

        # Check edges and  below as well
        for (dy, dx) in DIRECTIONS:
            ny, nx = y + dy, x + dx
            if is_valid(nx, ny, prev_slic):

                if dy != 0 and dx != 0:
                    support_score += CORNER_SUPPORT_VALUE

                else:
                    support_score += EDGE_SUPPORT_VALUE

    # No supports
    if support_score == 0:
       return False

    overhang_area = len(overhang)

    # Small overhangs only need 1 support, i.e. nonzero support score
    if overhang_area < MIN_AREA_TO_CONSIDER:
        return True

    # Large overhangs
    # return support_score >= required_support_score(overhang)
    return support_score >= overhang_area * REQUIRED_SUPPORT_FACTOR


def get_overhang_mask(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Return boolean mask with same shape as input grid where True
    represents an overhang.
    """

    size_z, size_y, size_x = grid.shape
    prev_slic = grid[0]
    overhang_mask = np.zeros_like(grid, dtype=bool)
    n_overhangs = 0
    for z in range(1, size_z):
        slic = grid[z]
        visited = np.zeros_like(slic, dtype=bool)
        
        for y in range(size_y):
            for x in range(size_x):
                if slic[y][x] == 1 and not visited[y][x]:
                    overhang = bfs_2d(x, y, slic, visited)

                    if not is_supported(overhang, prev_slic):
                        n_overhangs += 1
                        for ox,oy in overhang:
                            overhang_mask[z,oy,ox] = True
    
        prev_slic = slic

    return overhang_mask, n_overhangs

def get_overhang_pts(grid: np.ndarray) -> np.ndarray:
    """Get all overhangs as Nx3 array of z,y,x points."""

    return np.transpose(np.nonzero(get_overhang_mask(grid)[0]))

def count_overhang_pts(grid: np.ndarray) -> int:
    """Get count of all overhangs in grid."""

    return np.count_nonzero(get_overhang_mask(grid)[0])


if __name__ == "__main__":
    # Example 3x3 array, 1 overhang
    # grid = np.zeros((3,3,3))
        
    # grid[0,0,0] = 1 # base can never count as overhang
    # grid[1,0,0] = 1 # supported below
    # grid[2,2,2] = 1 # unsupported
    voxel  = sys.argv[1] if len(sys.argv) > 1 else 'voxel.npy'
    grid = np.load(voxel)
    # grid = np.load('voxel.npy')

    num_overhangs = get_overhang_mask(grid)[1]

    print(f'Total number of overhangs not supported by the layer below (including diagonals): {num_overhangs}')
