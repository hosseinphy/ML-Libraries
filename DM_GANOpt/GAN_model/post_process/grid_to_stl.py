import re
import numpy as np
import trimesh
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import ball, binary_closing, binary_opening
from . import SurfaceNet
from . import utils2  # requires keep_outside_connected_voids(void: 0/1) -> 0/1

# -------------------------------
# Mesh post-filter: keep only components
# that touch the simulation box boundary.
# -------------------------------

def _touches_boundary(m: trimesh.Trimesh, dims_xyz, scale, eps=1e-6):
    """
    True if any vertex lies on a domain boundary plane.
    dims_xyz: (X,Y,Z) AFTER rotation but BEFORE padding.
    scale: world units per voxel for exported mesh coords.
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
    Keep only mesh components that touch the domain boundary and have >= min_faces.
    If none qualify, keep the largest by face count (fallback).
    """
    comps = mesh.split(only_watertight=False)
    kept = [c for c in comps if len(c.faces) >= min_faces and _touches_boundary(c, dims_xyz, scale)]
    if not kept:
        kept = [max(comps, key=lambda c: len(c.faces))]
    return trimesh.util.concatenate(kept)

# -------------------------------
# Main conversion
# -------------------------------

# def convert_to_stl(file_dir):
#     """
#     Converts voxel grids (1=solid, 0=void) to fluid (void) STL surfaces using:
#       - SDF + SurfaceNets (no Gaussian)
#       - volumetric pocket removal (outside-connected void only)
#       - mesh cleanup + Taubin smoothing
#       - boundary-connected component filter
#     """
#     SCALE_FACTOR = 0.25   # world units per voxel
#     TAUBIN_ITERS = 12     # set 0 to disable smoothing; typical 8–20
#     ROTATE_AXES = (0, 2)  # set to None to skip

#     file_dir = Path(file_dir).resolve()
#     sn = SurfaceNet.SurfaceNets()

#     for file in file_dir.glob("voxel_structure_upscaled_*.npy"):
#         m = re.search(r"voxel_structure_upscaled_(\d+)\.npy", file.name)
#         if not m:
#             continue

#         iteration = int(m.group(1))
#         input_path  = file
#         output_path = file_dir / f"pfc_surface_{iteration}.stl"

#         try:
#             grid = np.load(input_path).astype(np.uint8)  # 1=solid, 0=void
#         except FileNotFoundError:
#             print(f"File not found: {input_path}")
#             continue


def convert_to_stl(file_dir):

    SCALE_FACTOR = 0.25   # world units per voxel
    #TAUBIN_ITERS = 0     # set 0 to disable smoothing; typical 8–20
    ROTATE_AXES = (0, 2)  # set to None to skip
    
    #scale_factor = 0.25  # For final STL scaling
    sn = SurfaceNet.SurfaceNets()

    file_dir = Path("./").resolve() 
    #file_dir = Path(file_dir).resolve()
    
    for file in file_dir.glob("voxel_structure_upscaled_*.npy"):
        match = re.search(r"voxel_structure_upscaled_(\d+)\.npy", file.name)
        if match:
            iteration = int(match.group(1))
            print(iteration)
            input_path = file  # already a Path from glob
            output_path = file_dir / f"pfc_surface_{iteration}.stl"

            try:
                grid = np.load(input_path)
            except FileNotFoundError:
                print(f"File not found: {input_path}")
                continue  # skip and keep looping

        # --- target fluid: 1=void, 0=solid ---
        void = 1 - grid

        # --- remove disconnected fluid pockets (keep outside-connected only) ---
        void = utils2.keep_outside_connected_voids(void).astype(np.uint8)


        # --- morphological smoothing in voxel space (gentle, topology-aware) ---
        if True:
            # Rounds small jaggies: closing fills tiny gaps; opening removes tiny protrusions.
            se = ball(1)  # increase to 2 for stronger smoothing (may close thin channels)
            void = binary_closing(void, se)
            void = binary_opening(void, se)
            void = void.astype(np.uint8)

        # --- orient BEFORE padding/SDF so distances match final axes ---
        if ROTATE_AXES is not None:
            void = np.rot90(void, axes=ROTATE_AXES)

        # Save dims BEFORE padding for boundary test in world coords
        dims_xyz = void.shape  # (X,Y,Z) aligned with vertex axes

        # --- pad with voids to avoid boundary caps and keep outside connected ---
        # void = np.pad(void, 1, mode='constant', constant_values=1)
        void = np.pad(void, 1, mode='edge')

        # --- build SDF for fluid surface: >0 in void, <0 in solid; interface at 0 ---
        dist_void  = distance_transform_edt(void == 1)
        dist_solid = distance_transform_edt(void == 0)
        sdf = np.ascontiguousarray(dist_void - dist_solid)

        if False:
            # --- optional mild smoothing (e.g., 0.3–0.6) ---
            sdf_smooth = gaussian_filter(sdf, sigma=0.5)
            sdf = sdf_smooth

        # --- extract zero isosurface from SDF ---
        verts, faces = sn.surface_net(sdf, level=0.0)
        verts = np.asarray(verts, dtype=float)

        # --- undo +1 voxel padding shift in coords ---
        verts -= 1.0

        # --- scale to world units ---
        verts *= SCALE_FACTOR

        # --- build + clean mesh ---
        mesh = trimesh.Trimesh(verts, faces, process=False, validate=False)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()  # exact duplicates only

        # # --- Taubin smoothing (non-shrinking) ---
        # if TAUBIN_ITERS > 0:
        #     from trimesh.smoothing import filter_taubin
        #     filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=TAUBIN_ITERS)

        # --- keep only components touching the domain boundary ---
        mesh = filter_connected_to_boundary(mesh, dims_xyz=dims_xyz, scale=SCALE_FACTOR, min_faces=50)

        mesh.export(output_path)
        print(f"✔ saved {output_path}")

