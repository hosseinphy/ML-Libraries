Here’s a clear, self-contained report you can hand to a teammate (or your future self) that documents the whole pipeline and the algorithms behind it.

# Report: Voxel → STL (Fluid Surface) Pipeline

## Objective

Given a binary 3D voxel grid (`1 = solid`, `0 = void`), generate a **clean, CFD-ready STL** of the **fluid (void)** region that:

* excludes internal bubbles not connected to the exterior,
* avoids artificial boundary caps,
* preserves channels and thin features,
* removes stray mesh islands not touching the domain boundary.

## Data conventions

* Input voxel grid: `grid ∈ {0,1}` with **1 = solid, 0 = void**.
* We work mostly with a **void mask**: `void = 1 - grid` → **1 = void, 0 = solid**.
* Mesh coordinates are in **voxel units**, later scaled by `SCALE_FACTOR` to world units.

---

# Algorithms (building blocks)

## A) Outside-connected void filtering (3D flood fill)

**Goal:** keep only void voxels that are connected to the outside of the box; drop enclosed void pockets (resin traps).

**Idea:** run a flood fill (BFS/DFS) *from the padded outer shell* across `void == 1`; mark visited voxels as “outside-connected”.

**Steps:**

1. Temporarily **pad** the void volume with a 1-voxel halo of **1s (void)** to guarantee an “outside” seed.
2. Initialize a queue with **all outer shell cells** that are `void == 1`.
3. BFS with **6-connectivity** (±x, ±y, ±z) only.
4. After BFS, **strip padding** and keep `void & visited` as the cleaned void mask.

**Pseudocode:**

```text
pad = pad(void, 1, constant=1)            # seed outside with void
seen = zeros_like(pad, bool)
Q = deque(all outer shell coords with pad==1) ; mark seen
while Q:
  p = Q.pop()
  for n in N6(p):
    if pad[n]==1 and not seen[n]: seen[n]=True; Q.push(n)
clean = (pad==1) & seen
clean = unpad(clean)
```

**Why this matters:** prevents the mesher from seeing **internal bubbles**, which are unmeshable as fluid.

**Complexity:** O(N) over voxels.

---

## B) Voxel-space morphological smoothing

**Goal:** gently round stair-step edges **without** moving the geometry off the voxel grid or shrinking volume like Laplacian smoothing.

**Operators:**

* **Closing** `binary_closing(void, ball(r))` → fills small gaps/holes.
* **Opening** `binary_opening(…, ball(r))` → removes small spikes.

**Typical setting:** `r = 1` (1-voxel radius). Increase to 2 for stronger effect (careful: may close thin channels).

**Order used:** **closing → opening** (fill minor void defects, then shave tiny protrusions). Convert to/from boolean for correctness.

---

## C) Rotation (`np.rot90`)

**Goal:** orient the grid to your expected world axes before distance transforms and surface extraction.

**Call:** `np.rot90(a, k=1, axes=(0, 2))` rotates 90° CCW in the (Z,X) plane for each Y slice.
Pick the two axes that define the 2D plane you want to rotate.

---

## D) Padding for SDF / isosurface

**Goal:** stabilize the isosurface near the boundary **without** inventing artificial interfaces.

**Mode:** `mode='edge'` — replicates boundary values, so no new 0↔1 transitions are introduced.

**Note on coordinates:** padding adds a 1-voxel halo → the mesh vertices produced by SurfaceNets are in **padded** coordinates. **Shift** by −1 voxel per axis after extraction to map back to the original domain.

---

## E) Signed Distance Field (SDF)

**Goal:** convert binary `void/solid` to a **smooth scalar field** where the interface is the **zero level set**.

**Definition:**

* `dist_void  = EDT(void == 1)`
* `dist_solid = EDT(void == 0)`
* **Fluid SDF:** `sdf = dist_void - dist_solid`

  * `sdf > 0` inside void (fluid),
  * `sdf < 0` inside solid,
  * `sdf = 0` on the interface.

**Why SDF:** enables **sub-voxel accurate** vertex placement and smoother surfaces than extracting from a hard 0/1 field.

---

## F) Isosurface extraction (SurfaceNets)

**Goal:** extract the triangle mesh at `sdf = 0`.

**SurfaceNets basics:**

* Processes each active voxel cell once.
* Determines which edges cross the isovalue (here 0.0).
* Computes an interpolated **single vertex per cell** by averaging edge intersections (using a precomputed edge table).
* Connects cell vertices into faces consistently.

**Pros:** fewer skinny triangles; stable on noisy grids.
**Input:** contiguous array; we pass `np.ascontiguousarray(sdf)`.

---

## G) Coordinate shift & scaling

**Shift:** `verts -= 1.0` to undo the +1 halo offset.
**Scale:** `verts *= SCALE_FACTOR` to convert “voxels” to world units.

---

## H) Mesh cleanup

**Goal:** remove topological/degenerate artifacts so downstream meshing is robust.

**Ops (trimesh):**

* `remove_duplicate_faces()` — drop identical faces.
* `remove_degenerate_faces()` — remove zero-area / invalid triangles.
* `remove_unreferenced_vertices()` — drop orphaned vertices.
* `merge_vertices()` — merge **exact** duplicate vertices.

(If you ever need a tolerance-based weld, quantize positions or use a dedicated remesher—kept out here to avoid geometry drift.)

---

## I) Boundary-connected component filter (mesh-level)

**Goal:** remove any small, floating mesh **islands** produced by extraction that don’t touch the simulation box boundary.

**Method:**

1. **Split** the mesh into connected components.
2. **Keep** components with `faces ≥ min_faces` **and** that **touch** any domain face.
3. If none pass, keep the **largest** by face count.

**“Touches boundary” test:**

* Compute world-space bounds:

  * from **pre-pad** `dims_xyz` (after rotation) and `SCALE_FACTOR`
  * `x ∈ [0, (X-1)*scale]`, similarly for y, z
* Check if any vertex coordinate is `≈` (`np.isclose`) to min/max on any axis.

**Why:** guarantees the exported fluid surface is the **exterior-connected** fluid, not stray islands.

---

# End-to-end algorithm of `convert_to_stl`

Below is the **logical flow** your function implements (with the minor best-practice tweaks):

1. **Load** `grid` (`1=solid, 0=void`) from `voxel_structure_upscaled_*.npy`.
2. **Make void mask:** `void = 1 - grid` (`1=void, 0=solid`).
3. **Outside-connected void only:** `void = keep_outside_connected_voids(void)`.
4. **Morphological smoothing (voxel space):**

   * `void = binary_closing(void, ball(1))`
   * `void = binary_opening(void, ball(1))`
5. **Rotate** to desired orientation: `void = np.rot90(void, axes=(0, 2))`.
6. **Record domain size** pre-pad: `dims_xyz = void.shape`.
7. **Pad** for SDF stability (no new interfaces): `void = np.pad(void, 1, mode='edge')`.
8. **Build SDF (fluid):**

   * `dist_void  = EDT(void == 1)`
   * `dist_solid = EDT(void == 0)`
   * `sdf = dist_void - dist_solid`
   * `sdf = np.ascontiguousarray(sdf)`
9. **Surface extraction:** `(verts, faces) = SurfaceNets(sdf, level=0.0)`.
10. **Undo padding shift:** `verts -= 1.0`.
11. **Scale to world units:** `verts *= SCALE_FACTOR`.
12. **Mesh cleanup (trimesh):**

    * `remove_duplicate_faces()`
    * `remove_degenerate_faces()`
    * `remove_unreferenced_vertices()`
    * `merge_vertices()`
13. **Boundary component filter:**

    * `mesh = filter_connected_to_boundary(mesh, dims_xyz, SCALE_FACTOR, min_faces=50)`
14. **Export** STL.

---

# Parameters & tuning guide

* **Structuring element radius** (`ball(1)`): 1 is mild; 2 is stronger (can close thin channels).
* **Rotate axes** (`(0, 2)`): align with your downstream coordinate system.
* **Padding mode**:

  * For flood-fill (inside that function): `constant_values=1` (void) so outside is a seed.
  * For SDF: `mode='edge'` to avoid creating an artificial interface on the box face.
* **Boundary filter**:

  * `min_faces=50`: raise if tiny slivers remain; lower if you risk dropping valid thin sheets.
  * `eps=1e-6` in `isclose`: tolerance for plane tests (in **scaled** units).

---

# Common pitfalls & quick checks

* **Wrong convention into flood-fill:** ensure `keep_outside_connected_voids` expects **1=void**. If it expects 1=solid, invert in/out accordingly.
* **Forgetting the vertex shift after padding:** if STL looks offset by 1 voxel, you didn’t do `verts -= 1.0`.
* **Boundary caps in STL:** if you see a closed shell at the box, you padded SDF with `constant` instead of `edge`, or extracted from a binary field directly.
* **Over-smoothing (mesh):** Taubin/Laplacian can stretch channels. Prefer **voxel morphology** + SDF; skip mesh smoothing if channel fidelity matters.

---

# Minimal annotated code skeleton

```python
# 1) Load grid
grid = np.load(input_path).astype(np.uint8)     # 1=solid, 0=void

# 2) Void mask
void = 1 - grid                                 # 1=void, 0=solid

# 3) Drop internal bubbles
void = utils2.keep_outside_connected_voids(void).astype(np.uint8)

# 4) Morphological smoothing (voxel-space)
void = void.astype(bool)
void = binary_closing(void, ball(1))
void = binary_opening(void, ball(1))
void = void.astype(np.uint8)

# 5) Rotate (before distances)
void = np.rot90(void, axes=(0, 2))
dims_xyz = void.shape                            # for boundary planes

# 6) Pad (edge) for stable SDF near box boundary
void = np.pad(void, 1, mode='edge')

# 7) SDF for fluid
dist_void  = distance_transform_edt(void == 1)
dist_solid = distance_transform_edt(void == 0)
sdf = np.ascontiguousarray(dist_void - dist_solid)

# 8) SurfaceNets extraction
verts, faces = sn.surface_net(sdf, level=0.0)
verts = np.asarray(verts, float)

# 9) Undo pad origin shift, then scale
verts -= 1.0
verts *= SCALE_FACTOR

# 10) Mesh cleanup
mesh = trimesh.Trimesh(verts, faces, process=False, validate=False)
mesh.remove_duplicate_faces()
mesh.remove_degenerate_faces()
mesh.remove_unreferenced_vertices()
mesh.merge_vertices()

# 11) Keep boundary-connected components
mesh = filter_connected_to_boundary(mesh, dims_xyz, scale=SCALE_FACTOR, min_faces=50)

# 12) Export
mesh.export(output_path)
```

---

# TODO:
I need to implement an algorithm that can set up some seeds for open channels, to make the cfd meshing more accurate.
