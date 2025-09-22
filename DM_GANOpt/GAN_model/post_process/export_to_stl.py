from collections import defaultdict

import numpy as np
import trimesh
from stl import mesh
from skimage import measure

import utils2
import SurfaceNet

# grid = np.load('repaired.npy')
grid = np.pad(utils2.add_frame(np.load('voxel_structure.npy'), width=1), 1, 'constant', constant_values=0)
# grid = utils.add_frame(np.zeros((80,80,80)))

# Use marching cubes to obtain the surface mesh of these ellipsoids
# verts, faces, normals, values = measure.marching_cubes(grid, 0)

# Create the mesh
# grid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

# surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
# surf_mesh.export('frame.stl')


# Surface Net Implementation
sn = SurfaceNet.SurfaceNets()
vertices,faces = sn.surface_net(grid*100, 50)

cube = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    
    for j in range(3):
        
        cube.vectors[i][j] = vertices[f[j]]

posElem = defaultdict(list)

for i in range(3):
    for vert in cube.vectors:
        posElem[i].append(vert[i])

#this prints the min/max value per dimension
for posidx in posElem:
    print(posidx, np.min(posElem[posidx]), np.max(posElem[posidx]))
    
cube.save('cube.stl')

print("STL file saved as output.stl")
