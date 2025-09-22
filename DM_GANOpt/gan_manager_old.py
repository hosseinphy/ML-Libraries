
from __future__ import annotations
import sys, os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path


# --- Put project root on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- Now import via the package name ---
from GAN_model.pretrained_voxel_gen import voxel_generator
from GAN_model.issues.repair_issues import remove_issues
from GAN_model.post_process.upscale_voxels import create_upscaled_voxel
from GAN_model.post_process.grid_to_stl import convert_to_stl 
from GAN_model.gan_train import run_training


PathLike = Union[str, Path]

class GANManager:

    def __init__(self):
        pass

    def generate_structure(self, outdir: PathLike) -> None:

        outdir = Path(outdir)
        print("Outdir: ", outdir)
        #outdir.mkdir(parents=True, exist_ok=True)
        
        #print(outdir)
        # 1) Generate voxel structures
        #voxel_generator(outdir)
        voxel_generator()

        # 2) Repair voxels
        remove_issues(outdir)

        # 3) Upscale voxels
        create_upscaled_voxel(outdir)

        # 4) Convert to STL
        convert_to_stl(outdir)
    
    def generator_train(self, data: list[dict]) -> None:
        
        run_training(data)
