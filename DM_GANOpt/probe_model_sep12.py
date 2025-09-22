from dotenv import dotenv_values
import json
#import numpy as np
from random import randint, uniform
import requests
import sympy

import time
from urllib.parse import urljoin

import tarfile

from pathlib import Path
import glob

import mimetypes
try:
    from sympy.printing.numpy import NumPyPrinter
except ImportError:
    # previous location
    from sympy.printing.pycode import NumPyPrinter

import os
import sys

#----------------------#
def process_cmdline():
    argList = sys.argv
    argc = len(argList)

    # expected: instance_uuid, batch_size
    if argc-1 != 2:
        raise Exception("run with incorrect number of arguments ({} != 2)".format(argc-1))

    return (argList[1], int(argList[2]))

def get_limits():
    # read configuration from config.json
    fname = "config.json"
    with open(fname) as f:
        config = json.load(f)

    # get variable names
    inp_names = config.get('scalar_inp', [])

    # get ranges
    domains = []
    for inp in inp_names:
        name = inp
        domain = tuple(config['inp_ranges'][inp])
        
	# Should have discrete type eventually, for now just use continuous
        domains.append({'name': name, 'domain': domain, 'type': 'continious'})

    return domains

def domain_type(domain):
    if all(isinstance(d, int) for d in domain): 
        return 'discrete'
    return 'continuous' 

def read_pending():
    # read configuration from config.json
    fname = "config.json"
    with open(fname) as f:
        config = json.load(f)

    # get variable names
    invar = config.get('scalar_inp', [])

    # get input values
    fname = "dataset.json"
    with open(fname) as f:
        dataset = json.load(f)

    inpar = []

    for exp in dataset.get('complete', []):
        for subset in exp.get('incomplete', []):
            ss_in = [ subset['inputs'][iv] for iv in invar ]
            inpar.append(ss_in)
        for subset in exp.get('proposed', []):
            ss_in = [ subset['inputs'][iv] for iv in invar ]
            inpar.append(ss_in)

    return inpar

def suggest_single(domain):

    pts = [uniform(*d['domain']) for d in domain]
   
    # Check for discrete vs. cont. once type is implemented
    #pts = []
    #for d in domain:
        #if d['type'] == 'discrete':
        #    pt = randint(*d['domain'])
        #else:
        #    pt = uniform(*d['domain'])
        #pts.append(pt)

    return pts

def suggest_multi(domain, batch_size):
    return [suggest_single(domain) for b in range(batch_size)]


def propose_with_files(
    url_propose,
    file_paths,
    domain,
    points,
    file_api_name=None,      # e.g. "pfc_surface" when file_paths is str/list
    use_brackets=False       # True -> send as files[pfc_surface][] style
):
    """
    file_paths can be:
      - str: a single file used for all points (requires file_api_name)
      - list[str]:
          * if len(list) == len(points): use the i-th file for the i-th point (requires file_api_name)
          * else: treat as MANY files for the SAME key and send them all for each point (requires file_api_name)
      - dict[str, str|list[str]]: mapping api_key -> file(s); all files sent for each point
    """
    # normalize
    if isinstance(file_paths, str):
        file_paths_norm = [file_paths]
        file_paths_mode = "single"
    elif isinstance(file_paths, dict):
        file_paths_norm = file_paths
        file_paths_mode = "dict"
    else:
        file_paths_norm = list(file_paths)
        file_paths_mode = "list"

    sess = requests.Session()
    responses = []

    for idx, pt in enumerate(points):
        scalar_inputs_list = [{"name": d["name"], "value": p} for d, p in zip(domain, pt)]
        data_propose = {
            "mode": "GANOpt",
            "scalar_inputs": json.dumps(scalar_inputs_list),
        }

        files = []
        open_handles = []

        def _append_file(key, path):
            fh = open(path, "rb")
            open_handles.append(fh)
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            files.append((key, (os.path.basename(path), fh, mime)))

        try:
            if file_paths_mode == "dict":
                # multiple api keys; each may have 1..N files
                for api_key, paths in file_paths_norm.items():
                    if isinstance(paths, str):
                        paths = [paths]
                    for p in paths:
                        key = f"files[{api_key}][]" if use_brackets and len(paths) > 1 else (
                              f"files[{api_key}]" if use_brackets else api_key)
                        _append_file(key, p)

            elif file_paths_mode == "list":
                if len(file_paths_norm) == len(points):
                    # one file per point
                    if not file_api_name:
                        raise ValueError("file_api_name is required when using one file per point.")
                    key = f"files[{file_api_name}]" if use_brackets else file_api_name
                    _append_file(key, file_paths_norm[idx])
                else:
                    # many files for the same api key (send all for each point)
                    if not file_api_name:
                        raise ValueError("file_api_name is required when sending many files for one key.")
                    for p in file_paths_norm:
                        key = f"files[{file_api_name}][]" if use_brackets else file_api_name
                        _append_file(key, p)

            else:  # "single"
                if not file_api_name:
                    raise ValueError("file_api_name is required when file_paths is a single string.")
                key = f"files[{file_api_name}]" if use_brackets else file_api_name
                _append_file(key, file_paths_norm[0])

            resp = sess.post(url_propose, data=data_propose, files=files)
            #print("STATUS:", resp.status_code)
            try:
                print("BODY:", resp.json())
            except Exception:
                print("BODY:", resp.text)

            responses.append(resp)
        finally:
            for fh in open_handles:
                try:
                    fh.close()
                except Exception:
                    pass

    return responses



#def fetch_task_status(status_url: str, token: str | None = None) -> dict:
#    """
#    Single GET to the task-status endpoint; returns the JSON payload:
#    {"state": "..."} or {"state":"SUCCESS","result":[...]} or {"state":"FAILURE","error":"..."}.
#    """
#    headers = {}
#    if token:
#        headers["Authorization"] = f"Bearer {token}"
#
#    r = requests.get(status_url, headers=headers, timeout=15)
#    r.raise_for_status()
#    return r.json()


# -------- helpers --------
#def wait_for_exp_list(status_url: str, token: str | None = None, timeout: int = 300, interval: float = 2.0):
#    deadline = time.time() + timeout
#    headers = {"Authorization": f"Bearer {token}"} if token else {}
#    while time.time() < deadline:
#        resp = requests.get(status_url, headers=headers, timeout=15)
#        if resp.status_code >= 400:
#            resp.raise_for_status()
#        try:
#            data = resp.json()
#        except ValueError:
#            data = {}
#        state = data.get("state")
#        if state == "SUCCESS":
#            return data.get("result", [])
#        if state in {"FAILURE", "REVOKED"}:
#            raise RuntimeError(data.get("error") or f"Task ended with state {state}")
#        retry_after = resp.headers.get("Retry-After")
#        time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else interval)
#    raise TimeoutError(f"Task did not finish within {timeout} seconds")



def wait_for_exp_list(
    status_url: str,
    token: str | None = None,
    timeout: int = 300,
    interval: float = 2.0,
    max_jitter: float = 0.5,
    raise_on_failure: bool = True,
):
    """
    Polls a Celery task status endpoint until SUCCESS/FAILURE/timeout.
    Returns the exp_list (from "result") on success.

    - Normalizes trailing slash to avoid Django 301s.
    - Sends Accept: application/json.
    - Respects Retry-After header if present.
    - Does not crash on non-JSON bodies; surfaces server error text.
    - If raise_on_failure=False, returns None on FAILURE/REVOKED.
    """
    # normalize URL (avoid 301)
    if not status_url.endswith("/"):
        status_url += "/"

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    deadline = time.time() + timeout

    while time.time() < deadline:
        resp = requests.get(status_url, headers=headers, timeout=15, allow_redirects=True)

        # Try JSON either way (even on 5xx)
        try:
            data = resp.json()
        except ValueError:
            data = {}

        state = data.get("state")

        # Success path
        if resp.status_code == 200 and state == "SUCCESS":
            return data.get("result", [])

        # In-progress path
        if resp.status_code in (200, 202) and state in {"PENDING", "STARTED", "RETRY"}:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_for = float(retry_after)
            else:
                sleep_for = interval + random.uniform(0, max_jitter)
            time.sleep(sleep_for)
            continue

        # Failure / revoked / unexpected
        err = data.get("error") or resp.text[:500] or f"HTTP {resp.status_code}"
        msg = f"Task {state or resp.status_code}: {err}"
        if raise_on_failure:
            raise RuntimeError(msg)
        return None

    raise TimeoutError(f"Task did not finish within {timeout} seconds")

    


if __name__ == "__main__":
    instance_uuid, batch_size = process_cmdline()
    env = dotenv_values()

    # ----- endpoints -----
    api_base = (env.get("dm_executor_url") or "").rstrip("/")
    if not api_base:
        raise RuntimeError("Missing dm_executor_url in .env")

    url_propose  = f"{api_base}/backend/{instance_uuid}/propose"
    url_complete = f"{api_base}/backend/{instance_uuid}/all_proposed"   # your named route
    #api_base_tasks = f"{api_base}/tasks"  # status endpoint base

    # Optional auth (set DM_AUTH_TOKEN in .env if you need it)
    headers = {}
    if env.get("DM_AUTH_TOKEN"):
        headers["Authorization"] = f"Bearer {env['DM_AUTH_TOKEN']}"

    # ----- files -----
    current_dir = os.path.abspath(os.path.dirname(__file__))

    file_list = [
        os.path.join(current_dir, "instances", instance_uuid, f"pfc_surface_{i}.stl")
        for i in range(batch_size)
    ]
    missing = [p for p in file_list if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing STL files: {missing[:3]}{'...' if len(missing) > 3 else ''}")

    # ----- points -----
    domain = get_limits()
    points = suggest_multi(domain, batch_size) if batch_size > 1 else [suggest_single(domain)]
    if len(points) != len(file_list):
        raise ValueError(f"points ({len(points)}) != files ({len(file_list)})")

  
    if False:
      # ----- send proposals: one request per (file, point) -----
        for fpath, pt in zip(file_list, points):
            # If propose_with_files expects a list of points, pass [pt]
            propose_with_files(url_propose, fpath, domain, [pt], file_api_name="pfc_surface")

        # ----- notify completion (queues Celery, returns task_id/status_url) -----
        resp = requests.post(url_complete, json={}, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        exp_list = data["experiments"]

        print("STATUS : ", resp.status_code)
        print("RAW : ", resp.text[:200])
        print("EXP LIST: ", exp_list)

    exp_list = ['967e860f-8dca-11f0-a068-274075a64213', 'dd1ef172-8f1f-11f0-a97e-274075a64213']


    ######################################################
    # Assign retrieved exp_uuid to the generated filenames
    ######################################################

    # files to use for training 
    train_files = [
        os.path.join(current_dir, "instances", instance_uuid, f"fake_voxel_batch_{i}.npy")
        for i in range(batch_size)
    ]

    latent_files = [
            os.path.join(current_dir, "instances", instance_uuid, "latents", f"fake_voxel_batch_{i}.latent.npy")
            ] 


    if False:
        compress = 'gz'

        # choose tarfile mode + extension
        mode_map = {None: "w", "gz": "w:gz", "bz2": "w:bz2", "xz": "w:xz"}
        ext_map  = {None: ".tar", "gz": ".tar.gz", "bz2": ".tar.bz2", "xz": ".tar.xz"}
        mode = mode_map.get(compress)
        ext  = ext_map.get(compress)
        if mode is None:
            raise ValueError("compress must be one of: 'gz', 'bz2', 'xz', or None")

        # put all the files under the experiment name (loop through files and experiment list)
        for i, e_uuid in enumerate(exp_list):
            print(f"i: {i}, exp_uuid: {e_uuid}")
            src = Path(train_files[i]).resolve(strict=True)
            filename = Path(file_list[i]).name
            arcname = str(e_uuid) +"_"+ filename
            arcpath = os.path.join(current_dir, "instances", instance_uuid, f"{arcname}{ext}")
            
            print("SRC: ", src)
            print("FILENAME: ", filename)
            print("ARC NAME: ", arcname)
            print("ARC PATH: ", arcpath)

            if not os.path.exists(arcpath):
                with tarfile.open(arcpath, mode) as tar:
                    tar.add(src, arcname=arcname)
            
            # now we want to also rename the .npz file (for latent vectors)
            #npz_file = os.path.join(current_dir, "instances", instance_uuid, "latents.npz")
            src = Path(latent_files[i]).resolve(strict=True)
            filename = Path(latent_files[i]).name
            new_name = str(e_uuid) + "_" + filename
            dst = src.with_name(new_name)  # same folder, new name
            src.rename(dst)                             # or: src.replace(dst) for overwrite



    ######################################################
    # Wait for CFD simulation results to continue training 
    ######################################################

    # check the results for the exps that have recieved

    # get input values
    fname = "dataset.json"
    with open(fname) as f:
        dataset = json.load(f)

    data_train = []

    for exp in dataset.get('complete', []):
        if exp.get('uuid') in exp_list:
            exp_uuid  = exp.get('uuid')
            actual = exp.get('actuals', '')
            if actual:
                so = actual[0].get('outputs', '')
                k, v = [list(so.keys())[0], list(so.values())[0]]
                # find a filenme
                outdir = Path(os.path.join(current_dir, "instances", instance_uuid)).resolve()
                for f in outdir.glob("*.tar.gz"):
                    if str(exp_uuid) in f.name:
                        fname = f.name

                data_train.append({str(exp_uuid) : {"file": os.path.join(outdir, fname), k: v }})


    print(data_train)

    # now complete the training 
#    if data_train:




