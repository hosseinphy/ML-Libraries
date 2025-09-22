import torch

import sys
from vogen import struct_gen

if __name__=="__main__":
    
    module_name, outdir = sys.argv[1], sys.argv[2]

    epochs = 1
    for epoch in epochs:
        
        # ------- Generate Random structures --------
        struct_gen(module_name, outdir)
    

        # ------------------- Physics Prediction -------------------
        # 1. Gather measurables (outputs of CFD Sims) & generated voxels --------
        # TODO: Check if the params has been recieved

            
        # 2. Calculate the physical loss





