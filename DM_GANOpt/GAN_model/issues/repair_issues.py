"""Attemps to repair all issues within a given voxel grid.
"""

import os
import numpy as np
import cv2

import sys
from . import overhangs
from . import islands
from . import resin_traps
from . import utils
import glob
import re
from pathlib import Path 
from collections import deque
from skimage.morphology import ball, binary_opening, binary_closing
from scipy.ndimage import label

from .functions.utils import plot_vtk


#current_dir = os.path.dirname(os.path.abspath(__file__))
#functions = os.path.join(current_dir, "functions")

#sys.path.append(functions)


# 6-connectivity mask for labeling
_CONN6 = np.zeros((3,3,3), dtype=np.uint8)
_CONN6[1,1,0] = _CONN6[1,1,2] = 1
_CONN6[1,0,1] = _CONN6[1,2,1] = 1
_CONN6[0,1,1] = _CONN6[2,1,1] = 1

def exterior_void_mask(void: np.ndarray) -> np.ndarray:
    """BFS 6-connected from boundary; void is bool (True=void)."""
    from collections import deque
    D,H,W = void.shape
    vis = np.zeros_like(void, dtype=bool)
    q = deque()
    def push(z,y,x):
        if 0<=z<D and 0<=y<H and 0<=x<W and void[z,y,x] and not vis[z,y,x]:
            vis[z,y,x] = True
            q.append((z,y,x))
    # enqueue all boundary voxels that are void
    for z in [0, D-1]:
        for y in range(H):
            for x in range(W): push(z,y,x)
    for z in range(D):
        for y in [0, H-1]:
            for x in range(W): push(z,y,x)
    for z in range(D):
        for y in range(H):
            for x in [0, W-1]: push(z,y,x)
    # BFS
    while q:
        z,y,x = q.popleft()
        for dz,dy,dx in ((0,0,-1),(0,0,1),(0,-1,0),(0,1,0),(-1,0,0),(1,0,0)):
            push(z+dz, y+dy, x+dx)
    return vis

def _articulation_filter(throat_mask: np.ndarray, interior_void: np.ndarray, window: int = 2) -> np.ndarray:
    """
    Keep only candidates whose removal increases connected-component count
    within a local window; reduces false positives.
    """
    keep = np.zeros_like(throat_mask, dtype=bool)
    coords = np.argwhere(throat_mask)
    D,H,W = throat_mask.shape
    for z,y,x in coords:
        z0, z1 = max(0, z-window), min(D, z+window+1)
        y0, y1 = max(0, y-window), min(H, y+window+1)
        x0, x1 = max(0, x-window), min(W, x+window+1)
        sub = interior_void[z0:z1, y0:y1, x0:x1].copy()

        # before
        _, n_before = label(sub, structure=_CONN6)
        # remove candidate
        sub[z - z0, y - y0, x - x0] = False
        _, n_after = label(sub, structure=_CONN6)

        if n_after > n_before:
            keep[z,y,x] = True
    return keep

def seal_internal_void_necks_minimal(grid: np.ndarray,
                                     min_width_vox: int = 1,
                                     verify_articulation: bool = True,
                                     window: int = 2) -> np.ndarray:
    """
    Seal ONLY tiny channels inside the structure (interior void), leaving
    exterior-connected void untouched. Modifies as little as possible:
    flips only the throat voxels from void->solid.

    grid: (D,H,W), 1=solid, 0=void
    """
    assert grid.ndim == 3
    solid = grid.astype(bool)
    void  = ~solid

    # 1) interior void only
    ext_void   = exterior_void_mask(void)
    interior   = void & (~ext_void)

    # 2) opening to *identify* ~1-voxel necks (on interior only)
    radius = max(1, int(np.ceil(min_width_vox / 2)))
    se = ball(radius)
    interior_open = binary_opening(interior, footprint=se)

    # 3) candidates = voxels removed by opening -> likely throats
    throat_candidates = interior & (~interior_open)

    # 4) optionally keep only "true necks" that split connectivity
    if verify_articulation:
        throat = _articulation_filter(throat_candidates, interior, window=window)
    else:
        throat = throat_candidates

    # 5) apply minimal edit: only flip those throat voxels to solid
    out = grid.copy()
    out[throat] = 1
    return out.astype(np.uint8)


def smooth(grid):
    kernel = np.ones((1, 1), np.uint8)
    for z,slic in enumerate(grid.astype('uint8')):
        slic = cv2.erode(slic, kernel, iterations=2)
        grid[z] = cv2.dilate(slic, kernel, iterations=2)

    return grid

def support_overhangs(grid, overhangs, connect_to_base=False):
    """Will produce a support below overhang.
    """
    for z,y,x in overhangs:
        if connect_to_base:
            grid[0:z,y,x] = 1
        
        # Add support below until nonzero value in grid
        else:
            i = 1
            while (z-i >= 0) and (grid[z-i,y,x] == 0):
                grid[z-i,y,x] = 1
                i += 1

    return grid
    


def erode_overhangs(grid: np.ndarray):
    """Iteratively removes all points marked as overhangs.
    """

    ohs_pts = overhangs.get_overhang_pts(grid)
    n_ohs = len(ohs_pts)

    while n_ohs > 0:
        for z,y,x in ohs_pts:
            grid[z,y,x] = 0
        ohs_pts = overhangs.get_overhang_pts(grid)
        n_ohs = len(ohs_pts)

    return grid


def remove_issues(outdir):

    outdir = Path("./").resolve() # Path(outdir).resolve()
    # outdir = Path(outdir).resolve()

    for file in outdir.glob('fake_voxel_batch_*.npy'):
        match = re.search(r"fake_voxel_batch_(\d+)\.npy", file.name)
        if match:
            iteration = int(match.group(1)) 

            voxel = outdir / file
            output_voxel = outdir / f"repaired_ohs_{iteration}.npy"
    
            grid = utils.add_frame(smooth(np.load(voxel)))

            # --- Remove islands ---
            ils, n_ils = islands.get_island_mask(grid)
            grid[ils] = 0
                
            # --- Remove overhangs ---    
            ohs_pts = overhangs.get_overhang_pts(grid)
            n_ohs = overhangs.count_overhang_pts(grid)
            grid = support_overhangs(grid, ohs_pts)

            # # New, trap-safe neck breaking
            # grid = break_thin_void_necks_no_traps(grid.astype(np.uint8), min_width_vox=2)
            occ_before = (grid>0).mean()
            
            rts, nrts = resin_traps.get_resin_trap_mask(grid)
            
            print(f"number of resin traps, including the first slice, BEFORE resolving tiny necks: {nrts}")
            print(f"number of overhang Before removing necks: {n_ohs}")
            print(f"number of islands Before removing necks: {n_ils}")

            # --- Seal *internal* micro-channels only ---
            grid = seal_internal_void_necks_minimal(grid, min_width_vox=1, verify_articulation=True, window=2)
            occ_after  = (grid>0).mean()
            print("Δ solid occupancy:", occ_after - occ_before)


            # Repair resin traps by filling in holes
            rts, nrts = resin_traps.get_resin_trap_mask(grid)
            # Don't repair in first slices
            rts[:, :, 0:1] = 0
            # Apply fix
            grid[rts == 1] = 1

            # check issues after applying voxel neck break
            ils, n_ils = islands.get_island_mask(grid)
            n_ohs = overhangs.count_overhang_pts(grid)


            print(f"number of resin traps, including the first slice, AFTER resolving tiny necks: {nrts}")
            print(f"number of overhang After removing necks: {n_ohs}")
            print(f"number of islands After removing necks: {n_ils}")

            
            np.save(output_voxel, utils.remove_frame(grid))



