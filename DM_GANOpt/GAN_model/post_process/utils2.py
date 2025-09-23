import numpy as np
import trimesh


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



def _touches_boundary(m: trimesh.Trimesh, dims_xyz, scale, eps=1e-6):
    """
    Returns True if any vertex of mesh 'm' lies on a domain boundary plane.

    dims_xyz: (X, Y, Z) voxel counts AFTER rotation but BEFORE padding.
    scale: voxel size -> world units used for exported mesh coords.
    """
    X, Y, Z = dims_xyz
    x_min, x_max = 0.0, (X - 1) * scale
    y_min, y_max = 0.0, (Y - 1) * scale
    z_min, z_max = 0.0, (Z - 1) * scale

    v = m.vertices
    on_xmin = np.any(np.isclose(v[:, 0], x_min, atol=eps))
    on_xmax = np.any(np.isclose(v[:, 0], x_max, atol=eps))
    on_ymin = np.any(np.isclose(v[:, 1], y_min, atol=eps))
    on_ymax = np.any(np.isclose(v[:, 1], y_max, atol=eps))
    on_zmin = np.any(np.isclose(v[:, 2], z_min, atol=eps))
    on_zmax = np.any(np.isclose(v[:, 2], z_max, atol=eps))
    return on_xmin or on_xmax or on_ymin or on_ymax or on_zmin or on_zmax


def filter_connected_to_boundary(mesh: trimesh.Trimesh, dims_xyz, scale, min_faces=50):
    """
    Split 'mesh' into connected components and keep only those that
    touch the domain boundary and are reasonably sized.

    If none pass, keep the largest component as a fallback.
    """
    comps = mesh.split(only_watertight=False)
    kept = []
    for c in comps:
        if len(c.faces) >= min_faces and _touches_boundary(c, dims_xyz, scale):
            kept.append(c)
    if not kept:
        kept = [max(comps, key=lambda c: len(c.faces))]
    return trimesh.util.concatenate(kept)


def keep_outside_connected_voids(void_grid: np.ndarray) -> np.ndarray:
    from collections import deque

    pad = np.pad(void_grid, 1, mode='constant', constant_values=1 )


    D, H, W = pad.shape
    vis = np.zeros_like(pad, dtype=bool)
    q = deque()

    def push(z, y, x):
        if 0<=z<D and 0<=y<H and 0<=x<W and pad[z, y, x] == 1 and not vis[z,y,x] :
            vis[z, y, x] = True
            q.append((z,y,x))

    for z in (0, D-1):
        for y in range(H):
            for x in range(W):
                push(z, y, x)

    for z in range(D):
        for y in (0, H):
            for x in range(W):
                push(z, y, x)

    for z in range(D):
        for y in range(H):
            for x in (0, W-1):
                push(z, y, x)

    
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while q:
        z, y, x = q.popleft()
        for dz, dy, dx in dirs:
            nz, ny, nx = z+dz, y+dy, x+dx
            push(nz, ny, nx)

    outside_connected = np.logical_and(pad==1, vis)
    return outside_connected[1:-1, 1:-1, 1:-1].astype(np.uint8)

