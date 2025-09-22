
import sys, os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pretrained_voxel_gen import voxel_generator

# Add issues directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
issues_dir = os.path.join(current_dir, "issues")
postprocess_dir = os.path.join(current_dir, "post_process")
sys.path.append(issues_dir)
sys.path.append(postprocess_dir)

from issues.repair_issues import remove_issues
from post_process.upscale_voxels import create_upscaled_voxel
from post_process.grid_to_stl import convert_to_stl 


def struct_gen(module_name: str, outdir: str):


     # ------- Create voxel structures --------
    voxel_generator(outdir)

    # ------- Repair voxels --------
    remove_issues(outdir)

    # ------- Upscale voxels --------
    create_upscaled_voxel(outdir)

    # ------- Convert grid to stl --------
    convert_to_stl(outdir)






