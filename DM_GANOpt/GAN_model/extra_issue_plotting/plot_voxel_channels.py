import numpy as np
from collections import deque

# plot open vs dead-end channels
# 
# 
# # first find out whcih channel is external and which is internal andlabel them 
def  ext_void_mask(void: np.ndarray)-> np.ndarray:
    D, H, W = void.shape 
    vis = np.zeros_like(void, dtype=bool)
    q = deque()

    def push(z, y, x):
        if 0<=z<D and 0<=y<H and 0<=x<W and void[x, y, z] and not vis[x, y, z]:
            vis[x, y, z] = 1
            q.append((x, y, z))
    
    for z in [0, D-1]:
        for y in range(H):
            for x in range(W):
                push(z,y,x)

    for z in range(D):
        for y in [0, H-1]:
            for x in range(W):
                push(z,y,x)


    for z in range(D):
        for y in range(H):
            for x in [0, W-1]:
                push(z,y,x)

    return vis

