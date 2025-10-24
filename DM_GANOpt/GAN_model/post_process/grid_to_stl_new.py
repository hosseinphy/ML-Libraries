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

            # --- LOAD & PREPARE FIELD -------------------------------------------------
            grid = np.asarray(grid)

            # squeeze to 3D if a singleton channel slipped in
            if grid.ndim == 4:
                grid = np.squeeze(grid)
            if grid.ndim != 3:
                print(f"Skipping {file}: expected 3D, got {grid.shape}")
                continue

            # Ensure binary in {0,1} if floats are present
            mn, mx = float(grid.min()), float(grid.max())
            if not (mn >= 0.0 and mx <= 1.0):
                rng = (mx - mn) or 1.0
                grid = (grid - mn) / rng
            grid = (grid >= 0.5).astype(np.uint8)

            # Decide which phase to surface: prefer the one that isn't uniform
            def non_uniform(a): 
                return not (a.min() == a.max())

            solid = grid              # 1=solid, 0=void
            void  = 1 - grid          # 1=void,  0=solid

            if non_uniform(void):
                target = void         # Extract the *fluid* surface (often what you want for CFD)
                pad_val = 1           # keep outside connected in padding
                level  = 0.5          # for marching_cubes fallback
                which  = "void"
            elif non_uniform(solid):
                target = solid        # Extract *solid* surface
                pad_val = 0
                level  = 0.5
                which  = "solid"
            else:
                print(f"Uniform grid ({mn}); no surface to extract. Skipping.")
                continue

            # Rotate to orient correctly (if you need this convention)
            target = np.rot90(target, axes=(0, 2))

            # Pad so the surface closes properly; pad with the phase we keep connected
            target = np.pad(target, 1, mode="constant", constant_values=pad_val)

            # --- PRIMARY: SDF + SurfaceNets -------------------------------------------
            # Build a signed distance: >0 in target-phase, <0 in the other
            from scipy.ndimage import distance_transform_edt
            dist_in   = distance_transform_edt(target == 1)
            dist_out  = distance_transform_edt(target == 0)
            sdf = dist_in - dist_out

            verts, faces = sn.surface_net(sdf, level=0.0)  # zero-crossing of SDF

            # --- If SurfaceNets failed, fallback to marching_cubes on the chosen phase ---
            if len(faces) == 0:
                try:
                    from skimage import measure
                    v2, f2, *_ = measure.marching_cubes(target.astype(np.float32), level=0.5)
                    # marching_cubes returns (z,y,x); swap z<->x to mimic rot90 over (0,2)
                    v2 = v2.astype(np.float64)
                    v2[:, [0, 2]] = v2[:, [2, 0]]
                    verts, faces = v2, f2
                    print(f"Fallback: marching_cubes on {which}.")
                except Exception as e:
                    print(f"No surface from SurfaceNets and fallback failed: {e}. Skipping.")
                    continue

            # --- COERCE FACES TO TRIANGLES (SurfaceNets may output quads) --------------
            def _faces_to_triangles(faces):
                import numpy as np
                if isinstance(faces, np.ndarray) and faces.ndim == 2:
                    polys = faces.tolist()
                else:
                    polys = [list(map(int, f)) for f in faces]
                cleaned = [p for p in polys if len(p) in (3, 4)]
                if not cleaned:
                    return np.empty((0, 3), dtype=np.int64)
                f2 = np.array(cleaned, dtype=np.int64)
                if f2.shape[1] == 3:
                    return f2
                t1 = f2[:, [0, 1, 2]]
                t2 = f2[:, [0, 2, 3]]
                return np.vstack([t1, t2]).astype(np.int64)

            verts = np.asarray(verts, dtype=np.float64)
            faces = _faces_to_triangles(faces)
            if faces.size == 0:
                print("No triangular faces after coercion; skipping.")
                continue

            # Undo the +1 pad in world coords
            verts -= 1.0

            # Scale to world units
            verts *= scale_factor

            # --- BUILD & EXPORT --------------------------------------------------------
            surf = trimesh.Trimesh(verts, faces, process=False, validate=False)

            if len(surf.faces) == 0:
                print("Mesh has 0 faces after construction; skipping export.")
                continue

            surf.remove_duplicate_faces()
            surf.remove_degenerate_faces()
            surf.remove_unreferenced_vertices()
            surf.merge_vertices()

            if len(surf.faces) == 0:
                print("Mesh became empty after cleanup; skipping export.")
                continue

            surf.export(output_path)
            print(f"Saved {output_path}")





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

