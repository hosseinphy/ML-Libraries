
import requests
import json
import mimetypes
import os



#def propose_with_files(url_propose, file_paths, domain, points, file_api_name="input"):
#    if isinstance(file_paths, str):
#        file_paths = [file_paths]
#    else:
#        file_paths = list(file_paths)
#
#    reuse_single_file = (len(file_paths) == 1)
#    responses = []
#
#    for idx, pt in enumerate(points):
#        scalar_inputs_list = [{"name": d["name"], "value": p} for d, p in zip(domain, pt)]
#
#        data_propose = {
#            "mode": "GANOpt",
#            # JSON string is OK; DRF JSONField will parse it in multipart
#            "scalar_inputs": json.dumps(scalar_inputs_list),
#        }
#
#        path = file_paths[0] if reuse_single_file else file_paths[idx]
#        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
#
#        # IMPORTANT: use a simple field name that your serializer defines, e.g. "input"
#        with open(path, "rb") as f:
#            files = { "inputs": (os.path.basename(path), f, mime) }
#            resp = requests.post(url_propose, data=data_propose, files=files)
#
#        print("STATUS:", resp.status_code)
#        try:
#            print("BODY:", resp.json())
#        except Exception:
#            print("BODY:", resp.text)
#
#        responses.append(resp)
#
#    return responses



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




# Endpoint URLs
instance_uuid = "da66ce77-8382-11f0-87e9-13157b2dc4a5"
api_base="http://132.156.102.203/MAPS/stable/ccusdm/decision_maker/backend"
url_propose = f"{api_base}/{instance_uuid}/propose"
#url_complete = f"{api_base}/{instance_uuid}/all_proposed"

# files
file_list = ['/home/maps/DM_GANOpt/instances/da66ce77-8382-11f0-87e9-13157b2dc4a5/pfc_surface_0.stl']

# domain
domain = []

# Generate points
points = [[]]

# Send: one request per (point, file_i)
propose_with_files(url_propose, file_list[0], domain, points, file_api_name="pfc_surface")



