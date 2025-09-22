from dotenv import dotenv_values
import json
#import numpy as np
from random import randint, uniform
import requests
import sympy

import time
from urllib.parse import urljoin

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
            print("STATUS:", resp.status_code)
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



def fetch_task_status(status_url: str, token: str | None = None) -> dict:
    """
    Single GET to the task-status endpoint; returns the JSON payload:
    {"state": "..."} or {"state":"SUCCESS","result":[...]} or {"state":"FAILURE","error":"..."}.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(status_url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def wait_for_exp_list(status_url: str, token: str | None = None, timeout: int = 300, interval: float = 2.0):
    """
    Polls the status endpoint until SUCCESS or FAILURE/REVOKED or timeout.
    Respects Retry-After header if present. Returns exp_list on success.
    """
    deadline = time.time() + timeout
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    while time.time() < deadline:
        resp = requests.get(status_url, headers=headers, timeout=15)
        # If your view returns 202 for in-progress, don't raise on it
        if resp.status_code >= 400:
            resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError:
            data = {}

        state = data.get("state")
        if state == "SUCCESS":
            # result is your exp_list
            return data.get("result", [])
        if state in {"FAILURE", "REVOKED"}:
            raise RuntimeError(data.get("error") or f"Task ended with state {state}")

        # Backoff/poll interval; respect Retry-After if provided
        retry_after = resp.headers.get("Retry-After")
        sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else interval
        time.sleep(sleep_for)

    raise TimeoutError(f"Task did not finish within {timeout} seconds")



if __name__ == "__main__":
    
    instance_uuid, batch_size = process_cmdline()
    env_confs = dotenv_values()

    domain = get_limits()  # ('continuous' typo doesn't break but you can fix it)

    # Build one STL path per point
    current_dir = os.path.abspath(os.path.dirname(__file__))
    file_list = [
        os.path.join(current_dir , "instances", instance_uuid, f"pfc_surface_{i}.stl")
        for i in range(batch_size)
    ]
    # If your instances/ directory is at the project root (not inside GAN_model), use:
    # file_list = [
    #     os.path.join(current_dir, "instances", instance_uuid, f"pfc_surface_{i}.stl")
    #     for i in range(batch_size)
    # ]

    #print(domain)
    # Generate points
    points = suggest_multi(domain, batch_size) if batch_size > 1 else [suggest_single(domain)]

    #print(points)
    # Endpoint URLs
    api_base = env_confs.get("dm_executor_url")
    api_base_tasks = env_confs.get("dm_executor_url_tasks")
    url_propose = f"{api_base}/{instance_uuid}/propose"
    url_complete = f"{api_base}/{instance_uuid}/all_proposed"

    
    # Send: one request per (point, file_i)
    propose_with_files(url_propose, file_list[0], domain, points, file_api_name="pfc_surface")


    # Notify completion
    res_data = requests.post(url_complete)


    resp = requests.post(url_complete)#, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()

    # Prefer JSON body
    task_id = None
    status_url = None
    try:
        data = resp.json()
        task_id = data.get("task_id")
        status_url = data.get("status_url")
    except ValueError:
        data = {}

    # Fallback to Location header
    if not status_url:
        status_url = resp.headers.get("Location")

    # Last resort: construct from task_id if route is known
    if not status_url and task_id:
        status_url = urljoin(api_base_tasks, f"/{task_id}/")

    if not status_url:
        raise RuntimeError("Could not determine status_url from response.")

    #return task_id, status_url


    print("Queued task:", task_id)
    print("Status URL:", status_url)

    # 2) poll until done and get exp_list
    exp_list = wait_for_exp_list(status_url, token=None, timeout=300, interval=2.0)
    print("exp_list:", exp_list)


    #
    #print(res_data)
    # 
    #

    #task_id = res_data["task_id"]  # from your POST response's task_id
    #
    #r = requests.get(f"{api_base_tasks}/{task_id}/", timeout=15)
    #r.raise_for_status()
    #print(r.json())

