import re
import numpy as np
import trimesh
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import ball, binary_closing, binary_opening
from . import SurfaceNet
from . import utils2  # requires keep_outside_connected_voids(void: 0/1) -> 0/1


# -------------------------------
# Main conversion
# -------------------------------

def convert_to_stl(file_dir):

    SCALE_FACTOR = 0.25   # world units per voxel
    #TAUBIN_ITERS = 0     # set 0 to disable smoothing; typical 8–20
    MORPH_SMOOTH = False
    GAUSSIAN_SMOOTH  = False
    ROTATE_AXES = (0, 2)  # set to None to skip
    
    sn = SurfaceNet.SurfaceNets()

    file_dir = Path("./").resolve() 
    # file_dir = Path(file_dir).resolve()
    
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

        # -- Sanity check ----
        if False:
            # Before vs after counts
            print("void voxels before:", int((1-grid).sum()))
            print("void voxels after :", int(void.sum()))

            # Ensure at least one boundary voxel remains void (i.e., connected to outside)
            print("any boundary void after?",
                bool(void[0].any() or void[-1].any() or
                    void[:,0,:].any() or void[:,-1,:].any() or
                    void[:,:,0].any() or void[:,:,-1].any()))


        # --- morphological smoothing in voxel space (gentle, topology-aware) ---
        if MORPH_SMOOTH:
            # Rounds small jaggies: closing fills tiny gaps; opening removes tiny protrusions.
            se = ball(1)  # increase to 2 for stronger smoothing (may close thin channels)
            void = void.astype(bool)    # cast as boolean as expected by binary opening/closing  
            void = binary_closing(void, se)
            void = binary_opening(void, se)
            void = void.astype(np.uint8) # revert back to uint8    

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

        if GAUSSIAN_SMOOTH:
            # --- optional mild smoothing (e.g., 0.3–0.6) ---
            sdf_smooth = gaussian_filter(sdf, sigma=0.5)
            sdf = sdf_smooth

        # --- extract zero isosurface from SDF ---
        verts, faces = sn.surface_net(sdf, level=0.0)
        if len(faces) == 0:
            print(f"No faces extracted for {input_path}, skipping")
            continue
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
        mesh = utils2.filter_connected_to_boundary(mesh, dims_xyz=dims_xyz, scale=SCALE_FACTOR, min_faces=50)

        mesh.export(output_path)
        print(f"saved {output_path}")

    
    # Do not delete these file -- for testing
    if False:
        # file_dir is a Path
        patterns = ("voxel_structure_upscaled_*.npy", "repaired_ohs_*.npy")
        files = [p for pat in patterns for p in file_dir.glob(pat)]

        for p in files:
            try:
                if p.is_file():
                    p.unlink()
            except OSError as e:
                print(f"Failed to delete {p}: {e}")


