#!/usr/bin/env python3
from __future__ import annotations

import requests 
import os, sys, json, time, random, tarfile, mimetypes
from pathlib import Path
from typing import Optional, List
from dotenv import dotenv_values
from gan_manager import GANManager

# ---------------------- #
# Basic CLI
# ---------------------- #
def process_cmdline():
    argList = sys.argv
    argc = len(argList)
    # expected: instance_uuid, batch_size
    if argc - 1 != 2:
        raise SystemExit(f"Usage: {argList[0]} <instance_uuid> <batch_size>")
    return (argList[1], int(argList[2]))

# ---------------------- #
# If you still need domain utils later (not used below)
# ---------------------- #
def get_limits() -> list[dict]:
    with open("config.json") as f:
        config = json.load(f)

    domains = []
    for name in config.get("scalar_inp", []):
        domain = tuple(config["inp_ranges"][name])
        # keep the spelling consistent
        domains.append({"name": name, "domain": domain, "type": "continuous"})
    return domains

def suggest_single(domain: list[dict]) -> list[float]:
    from random import uniform
    return [uniform(*d["domain"]) for d in domain]

def suggest_multi(domain: list[dict], batch_size: int) -> list[list[float]]:
    return [suggest_single(domain) for _ in range(batch_size)]

# ---------------------- #
# Renaming + tarring
# ---------------------- #
def rename_voxel_and_latent_for_batch(
    exp_uuids: list[str],
    batch_size: int,
    voxel_base: str = "fake_voxel_batch_",
    latent_dir: str = "latents",
) -> tuple[list[Path], list[Path]]:
    """
    For each index i, we expect:
      voxel:  ./fake_voxel_batch_{i}.npy
      latent: ./latents/fake_voxel_batch_{i}.latent.npy

    We rename to:
      voxel:  ./{exp_uuid}_fake_voxel_batch_{i}.npy
      latent: ./latents/{exp_uuid}_fake_voxel_batch_{i}.latent.npy

    Returns (renamed_voxel_paths, renamed_latent_paths)
    """
    cwd = Path.cwd()
    lat_dir = cwd / latent_dir
    lat_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(exp_uuids), batch_size)
    renamed_voxels: List[Path] = []
    renamed_latents: List[Path] = []

    for i in range(n):
        u = exp_uuids[i]

        # --- voxel ---
        src_voxel = cwd / f"{voxel_base}{i}.npy"
        if not src_voxel.exists():
            raise FileNotFoundError(f"Missing voxel file: {src_voxel}")
        dst_voxel = cwd / f"{u}_{voxel_base}{i}.npy"
        if not dst_voxel.exists():
            src_voxel.rename(dst_voxel)
        renamed_voxels.append(dst_voxel)

        # --- latent ---
        src_latent = lat_dir / f"{voxel_base}{i}.latent.npy"
        if not src_latent.exists():
            raise FileNotFoundError(f"Missing latent file: {src_latent}")
        dst_latent = lat_dir / f"{u}_{voxel_base}{i}.latent.npy"
        if not dst_latent.exists():
            src_latent.rename(dst_latent)
        renamed_latents.append(dst_latent)

    return renamed_voxels, renamed_latents


def tar_each_voxel(
    voxel_paths: list[Path],
    compress: Optional[str] = "gz",  # 'gz' | 'bz2' | 'xz' | None
) -> list[Path]:
    """
    For each voxel .npy path, create a tarball in the same dir:
      <voxel_name>.tar.<ext>   (arcname = original filename)
    Returns list of created tar paths.
    """
    mode_map = {None: "w", "gz": "w:gz", "bz2": "w:bz2", "xz": "w:xz"}
    ext_map  = {None: ".tar", "gz": ".tar.gz", "bz2": ".tar.bz2", "xz": ".tar.xz"}
    if compress not in mode_map:
        raise ValueError("compress must be one of: None, 'gz', 'bz2', 'xz'")

    mode = mode_map[compress]
    ext  = ext_map[compress]

    tar_paths: List[Path] = []
    for p in voxel_paths:
        p = Path(p).resolve(strict=True)
        tar_path = p.parent / f"{p.name}{ext}"
        if not tar_path.exists():
            with tarfile.open(tar_path, mode) as tar:
                tar.add(p, arcname=p.name)
        tar_paths.append(tar_path)
    return tar_paths

# ---------------------- #
# Build data_train by exp_uuid
# ---------------------- #
def build_data_train_records(
    exp_uuids: list[str],
    voxel_tar_dir: Path | None = None,
    latent_dir: Path | None = None,
    dataset_path: Path | str = "dataset.json",
) -> list[dict]:
    """
    For each exp_uuid:
      - find tar:   <voxel_tar_dir>/<exp_uuid>_*.npy.tar.gz
      - find latent:<latent_dir>/<exp_uuid>_*.latent.npy
      - read 'dataset.json' to grab the first output key/value for that exp
    Returns list of dicts like:
      {"exp_uuid": ..., "voxel_tar": "...", "latent": "...", "<out_key>": value}
    """
    outdir = voxel_tar_dir or Path.cwd()
    latdir = latent_dir or (Path.cwd() / "latents")

    # load dataset once
    with open(dataset_path) as f:
        dataset = json.load(f)

    # index outputs by uuid
    outputs_by_uuid = {}
    for exp in dataset.get("complete", []):
        u = exp.get("uuid")
        actuals = exp.get("actuals") or []
        if not u or not actuals:
            continue
        outputs = actuals[0].get("outputs") or {}
        outputs_by_uuid[u] = outputs

    records: List[dict] = []
    for u in exp_uuids:
        # tar match
        tar_match = None
        for f in Path(outdir).glob(f"{u}_*.npy.tar.*"):
            tar_match = f
            break
        if tar_match is None:
            for f in Path(outdir).glob("*.tar.*"):
                if u in f.name:
                    tar_match = f
                    break

        # latent match
        latent_match = None
        for f in Path(latdir).glob(f"{u}_*.latent.npy"):
            latent_match = f
            break
        if latent_match is None:
            # fallback to <uuid>.latent.npy if you ever switch to that scheme
            cand = Path(latdir) / f"{u}.latent.npy"
            if cand.exists():
                latent_match = cand

        # outputs
        outputs = outputs_by_uuid.get(u, {})
        out_key, out_val = (None, None)
        if outputs:
            out_key, out_val = next(iter(outputs.items()))

        if tar_match is None:
            print(f"[warn] no tar found for exp {u} in {outdir}")
        if latent_match is None:
            print(f"[warn] no latent found for exp {u} in {latdir}")

        rec = {
            "exp_uuid": u,
            "voxel_tar": str(tar_match) if tar_match else None,
            "latent": str(latent_match) if latent_match else None,
        }
        if out_key is not None:
            rec[out_key] = out_val
        records.append(rec)

    return records


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





# ---------------------- #
# MAIN
# ---------------------- #
if __name__ == "__main__":

    # ----- flags -----
    test_flag = False

    instance_uuid, batch_size = process_cmdline()
    env = dotenv_values()  # keep if you later call the API; not used here

    # ----- endpoints -----
    api_base = (env.get("dm_executor_url") or "").rstrip("/")
    if not api_base:
        raise RuntimeError("Missing dm_executor_url in .env")

    url_propose  = f"{api_base}/{instance_uuid}/propose"
    url_complete = f"{api_base}/{instance_uuid}/all_proposed"   # your named route
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
  
    #if not test_flag:
    # ----- send proposals: one request per (file, point) -----
    for fpath, pt in zip(file_list, points):
        # If propose_with_files expects a list of points, pass [pt]
        propose_with_files(url_propose, fpath, domain, [pt], file_api_name="pfc_surface")

    # ----- notify completion (queues Celery, returns task_id/status_url) -----
    resp = requests.post(url_complete, json={}, headers=headers, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    exp_list = data["experiments"]

    print("STATUS : ", resp.status_code)
    print("RAW : ", resp.text[:200])
    print("EXP LIST: ", exp_list)

    #if False:
    ## ------------------------------------------------------------------
    ## 1) Replace this with the real exp_list you received from the API
    ## ------------------------------------------------------------------
    #    # exp_list = [
    #    #     "967e860f-8dca-11f0-a068-274075a64213",
    #    #     "dd1ef172-8f1f-11f0-a97e-274075a64213",
    #    # ]
    #
    #    exp_list = ['631ed069-8e82-11f0-a8f7-274075a64213', '967e860f-8dca-11f0-a068-274075a64213', 'b6e209ae-8db3-11f0-a097-274075a64213', 
    #    '88031ba6-8dc9-11f0-a69b-274075a64213', '7aa2b508-8dc8-11f0-82f9-274075a64213', '6d1d3217-8dc7-11f0-81ae-274075a64213', 
    #    '5f44e80e-8dc6-11f0-959a-274075a64213', '53be5b15-8dc5-11f0-8c92-274075a64213', '4745558d-8dc4-11f0-a056-274075a64213', 
    #    '3975ab52-8dc3-11f0-9d7a-274075a64213', '2e09d49b-8dc2-11f0-a3a8-274075a64213', '204384b0-8dc1-11f0-9750-274075a64213', 
    #    '127658ed-8dc0-11f0-9cf6-274075a64213', '049eea17-8dbf-11f0-8e19-274075a64213', 'f80ba9b7-8dbd-11f0-9a96-274075a64213', 
    #    'e9873f54-8dbc-11f0-892d-274075a64213'
    #    ]

    if len(exp_list) < batch_size:
        print(f"[warn] exp_list has {len(exp_list)} IDs but batch_size={batch_size}; proceeding with min length.")
    n = min(len(exp_list), batch_size)

    # ------------------------------------------------------------------
    # 2) Rename **both** voxels and latents to include exp_uuid
    #    (expects CWD = instances/<instance_uuid>/)
    # ------------------------------------------------------------------
    renamed_voxels, renamed_latents = rename_voxel_and_latent_for_batch(
        exp_uuids=exp_list[:n],
        batch_size=batch_size,
        voxel_base="fake_voxel_batch_",
        latent_dir="latents",
    )

    # ------------------------------------------------------------------
    # 3) Tar each renamed voxel
    # ------------------------------------------------------------------
    tar_paths = tar_each_voxel(renamed_voxels, compress="gz")


    # ------------------------------------------------------------------
    # 4) Build data_train list with tar + latent + first CFD output
    # ------------------------------------------------------------------
    data_train = build_data_train_records(
        exp_uuids=exp_list[:n],
        voxel_tar_dir=Path.cwd(),
        latent_dir=Path.cwd() / "latents",
        dataset_path="dataset.json",
    )

    #print(json.dumps(data_train, indent=2))


    # ------------------------------------------------------------------
    # 4) Train on data
    # ------------------------------------------------------------------
    gm = GANManager()
    gm.generator_train(data_train)
