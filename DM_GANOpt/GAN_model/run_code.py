import subprocess
import re
import sys, os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#from funcmodule import  isnumber, isnumberlist, read_settings, ext_uuid, sample_config, get_module_inputs


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

def runcode(cmd):
    """from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout = {}
    if False:
        while True:
            line = p.stdout.readline().decode('UTF-8')
            if line and not 'warning' in line.lower():
                print(line)
                k, v = re.sub( r"[\s]", r"", line.strip(), count=1).split(':')
                stdout.update({k:v})
            if line == '' and p.poll() != None:
                break
    return p.stderr, stdout 



#===============================
# writing to file
#===============================
def write_results(outputs, module):

    fname = "module_{}_results.json".format(module)

    with open(fname, "w") as f:
        json.dump( outputs, f )


def write_to_log(outs):

    fname = "aiida_metadata.json"

    with open(fname, "w") as f:
        json.dump( outs, f )


def write_samples(input_vars, module):

    fname = "samples.json"

    nsamples = input_vars[module+'_nsamples']
    sample = sample_config(nsamples)

    with open(fname, "w") as f:
        json.dump(obj=sample, fp=f)



#===============================
# main
#===============================
if __name__ == "__main__":

    module_name, outdir = sys.argv[1], sys.argv[2]

     # ------- Create voxel structures --------
    voxel_generator(outdir)

    # ------- Repair voxels --------
    remove_issues(outdir)

    # ------- Upscale voxels --------
    create_upscaled_voxel(outdir)

    # ------- Convert grid to stl --------
    convert_to_stl(outdir)






