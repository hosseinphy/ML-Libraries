import os
import numpy as np
import trimesh
from skimage import measure
from pathlib import Path
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

            # Mike's version    
            # # Invert the grid
            # grid = 1 - grid
            # # Pad for surface closure
            # grid = np.pad(grid, 1, 'constant', constant_values=0)

            # # Rotate to orient correctly
            # grid = np.rot90(grid, axes=(0, 2))

            # # SurfaceNet surface extraction
            # sn = SurfaceNet.SurfaceNets()
            # verts, faces = sn.surface_net(grid * 100, 50)
            # verts = np.array(verts)
            # verts *= scale_factor  # Final scaling for STL

            # # Build and export mesh
            # surf = trimesh.Trimesh(verts, faces, validate=True)
            # surf.export(output_path)

             # 1) Target the fluid: 1 = void, 0 = solid
            void = 1 - grid

            # 2) Keep only outside-connected voids (drop enclosed pockets)
            void = utils2.keep_outside_connected_voids(void).astype(np.uint8)

            # 3) Rotate (if needed) BEFORE padding/SDF (consistent distances)
            void = np.rot90(void, axes=(0, 2))

            # 4) PAD WITH VOIDS so the outside stays fluid (avoids boundary caps)
            #    Option A (exactly what you asked for): constant=1 (adds a ring of void)
            void = np.pad(void, 1, mode='constant', constant_values=1)
            #    (Alternative is mode='edge' to avoid introducing any new interface)

            # 5) Build SDF for fluid surface: >0 in void, <0 in solid; interface at 0
            dist_void  = distance_transform_edt(void == 1)
            dist_solid = distance_transform_edt(void == 0)
            sdf = dist_void - dist_solid
            sdf = np.ascontiguousarray(sdf)

            # 6) Extract zero isosurface from SDF
            verts, faces = sn.surface_net(sdf, level=0.0)
            verts = np.asarray(verts, dtype=float)

            # 7) Undo the +1 voxel padding shift in coordinates
            verts -= 1.0

            # 8) Final scaling + export
            verts *= scale_factor
            mesh = trimesh.Trimesh(verts, faces, validate=True)
            mesh.export(output_path)
            print(f"✔ saved {output_path}")


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

