
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
from GAN_model.issues.draw_issues import draw_issues
from GAN_model.gan_train import run_training

PathLike = Union[str, Path]



class GANManager:
    def __init__(self) -> None:
        pass

    def generate_structure(self, outdir: PathLike) -> None:
        """
        Assumes CWD == instances/<instance_uuid>/
        Writes/reads files relative to CWD.
        """
        #workdir = Path.cwd()
        workdir = Path(outdir)
        print("Working directory:", workdir)

        # 1) Generate voxels into CWD
        # voxel_generator()  # your version writes fake_voxel_batch_{i}.npy + latents/...

        # 2) Repair voxels in CWD
        remove_issues(workdir)

        # 3) Upscale voxels in CWD
        create_upscaled_voxel(workdir)

        # 4) Convert to STL in CWD
        convert_to_stl(workdir)

    def generator_train(self, data: list[dict]) -> None:
        """
        `data` should contain items like:
        {
          "exp_uuid": "...",
          "voxel_tar": "/abs/path/<uuid>_fake_voxel_batch_0.npy.tar.gz" (or .npy),
          "latent": "/abs/path/latents/<uuid>_fake_voxel_batch_0.latent.npy",
          "pdrop": 138.49
        }
        """
        run_training(data)

    def generator_draw_issues(self, voxel_path: PathLike) -> None:
        """
        Draw issues in the voxel file at `voxel_path`.
        Saves figure as issues.png in the same directory.
        """
        draw_issues(voxel_path)


