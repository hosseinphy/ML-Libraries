import numpy as np
import trimesh
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter
from . import SurfaceNet
from . import utils2
import re

def delete_regex(folder, pattern, recursive=False, dry_run=True):
    rx = re.compile(pattern)
    base = Path(folder)
    it = base.rglob("*") if recursive else base.glob("*")
    for p in it:
        if p.is_file() and rx.search(p.name):
            print(f"{'[DRY] ' if dry_run else ''}{p}")
            if not dry_run:
                p.unlink()


def convert_to_stl(file_dir):
    
    scale_factor = 0.25  # For final STL scaling

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
        
        # # 1) Target the fluid: 1 = void, 0 = solid
        # void = 1 - grid

        # # 2) Keep only outside-connected voids (drop enclosed pockets)
        # void = utils2.keep_outside_connected_voids(void).astype(np.uint8)

        # # 3) Rotate (if needed) BEFORE padding/SDF (consistent distances)
        # void = np.rot90(void, axes=(0, 2))

        # # 4) PAD WITH VOIDS so the outside stays fluid (avoids boundary caps)
        # #    Option A (exactly what you asked for): constant=1 (adds a ring of void)
        # # void = np.pad(void, 1, mode='constant', constant_values=0)
        # # void = np.pad(void, 1, mode='edge')
        # #    (Alternative is mode='edge' to avoid introducing any new interface)

        # void = np.pad(void, 1, mode='constant', constant_values=0)


        # # 5) Build SDF for fluid surface: >0 in void, <0 in solid; interface at 0
        # dist_void  = distance_transform_edt(void == 1)
        # dist_solid = distance_transform_edt(void == 0)
        # sdf = dist_void - dist_solid
        # sdf = np.ascontiguousarray(sdf)

        # # 5.1 smoothing 
        # sdf_smooth = gaussian_filter(sdf, sigma=0.4)
        
        # # 6) Extract zero isosurface from SDF
        
        # sn = SurfaceNet.SurfaceNets()
        # verts, faces = sn.surface_net(sdf_smooth, level=0.0)
        # verts = np.asarray(verts, dtype=float)

        # # 7) Undo the +1 voxel padding shift in coordinates
        # verts -= 1.0

        # # 8) Final scaling + export
        # verts *= scale_factor
        # mesh = trimesh.Trimesh(verts, faces, validate=True)
        # mesh.export(output_path)
        # print(f"✔ saved {output_path}")


        # --- target fluid: make void mask 1=void, 0=solid ---
        void = 1 - grid

        # --- remove disconnected void pockets volumetrically ---
        void = utils2.keep_outside_connected_voids(void).astype(np.uint8)

        # --- orient BEFORE padding/SDF so distances match final axes ---
        void = np.rot90(void, axes=(0, 2))

        # Save dims BEFORE padding (for boundary test in world units)
        dims_xyz = void.shape  # (X, Y, Z) matching vertex coord axes after rot90

        # --- pad with voids to keep outside connected; avoids boundary caps ---
        void = np.pad(void, 1, mode='edge')

        # --- build SDF for fluid surface: >0 in void, <0 in solid; interface at 0 ---
        dist_void  = distance_transform_edt(void == 1)
        dist_solid = distance_transform_edt(void == 0)
        sdf = dist_void - dist_solid
        sdf = np.ascontiguousarray(sdf)

        # --- optional mild smoothing (e.g., 0.3–0.6) ---
        sdf_smooth = gaussian_filter(sdf, sigma=0.6)

        # --- extract zero isosurface from SDF ---
        sn = SurfaceNet.SurfaceNets()
        verts, faces = sn.surface_net(sdf_smooth, level=0.0)
        verts = np.asarray(verts, dtype=float)

        # --- undo +1 voxel padding shift in coords ---
        verts -= 1.0

        # --- scale to world units ---
        verts *= scale_factor

        # --- build + clean mesh ---
        mesh = trimesh.Trimesh(verts, faces, process=False, validate=False)

        # Correct cleanup sequence (trimesh API)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()  # adjust epsilon if needed (e.g., 1e-6)

        # Optionally run trimesh's processing pipeline (includes validations)
        mesh.process(validate=True)

        # --- keep only components touching the domain boundary ---
        mesh = utils2.filter_connected_to_boundary(mesh, dims_xyz=dims_xyz, scale=scale_factor, min_faces=50)

        mesh.export(output_path)
        print(f"✔ saved {output_path}")
