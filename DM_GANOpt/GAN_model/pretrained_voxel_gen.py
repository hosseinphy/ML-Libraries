from pathlib import Path
import os, json, time, random, secrets, hashlib
from typing import Optional, List
import numpy as np
import torch


# ---------- saving utilities ----------

def save_latents_npz(Z: torch.Tensor, out_path: str | Path, ids: list[str], **arrays) -> int:
    """
    Save a batch of latents + aligned arrays to a single compressed .npz
    (path is relative to current working directory if given as a simple filename).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Z_np = Z.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    N = Z_np.shape[0]

    if len(ids) != N:
        raise ValueError(f"ids length {len(ids)} != {N}")
    for k, v in arrays.items():
        if len(v) != N:
            raise ValueError(f"{k} length {len(v)} != {N}")

    # Save ids as Unicode strings (avoids pickle)
    np.savez_compressed(out_path, Z=Z_np, ids=np.array(ids, dtype="U"), **arrays)
    return N


def save_latents_per_batch(
    Z: torch.Tensor,
    out_dir: str | Path = ".",                   # relative to CWD
    base_name: str = "fake_voxel_batch_",        # aligns with voxel filenames
    ckpt_path: str | Path | None = None,
) -> list[Path]:
    """
    Saves each z_i as latents/<base_name>{i}.latent.npy and writes latents_manifest.json in CWD (or out_dir).
    Returns list of saved *relative* paths (Path objects).
    """
    out_dir = Path(out_dir)
    lat_dir = out_dir / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)

    Z_np = Z.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    B = Z_np.shape[0]

    def sha256sum(p: Optional[Path]) -> Optional[str]:
        if not p:
            return None
        p = Path(p)
        if not p.exists():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    ckpt_hash = sha256sum(Path(ckpt_path)) if ckpt_path else None

    paths: List[Path] = []
    for i in range(B):
        p = lat_dir / f"{base_name}{i}.latent.npy"
        np.save(p, Z_np[i], allow_pickle=False)
        # store relative path from CWD for convenience
        paths.append(p)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latent_dim": int(Z_np.shape[1]),
        "count": B,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_sha256": ckpt_hash,
        "items": [{"index": i, "latent_path": str(paths[i])} for i in range(B)],
        "note": "Latents saved by index; rename/bind to exp_uuid after server returns uuids."
    }
    with (out_dir / "latents_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return paths


# ---------- voxel generation (writes to CWD) ----------

def voxel_generator(
    *,
    seed: int | None = None,
    deterministic: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Generate and save a batch of voxels from a pre-trained GAN generator.

    - Assumes the current working directory is: instances/<instance_uuid>/
    - Saves voxels as:           ./fake_voxel_batch_{i}.npy
    - Saves per-sample latents:  ./latents/fake_voxel_batch_{i}.latent.npy

    Returns (voxel_paths, latent_paths) as lists of relative strings.
    """
    from GAN_model.functions.utils import calc_occupancy, plot_vtk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seeding
    if seed is None:
        seed = secrets.randbits(64)
    print(f"[voxel_generator] seed={seed}")
    torch.manual_seed(seed)
    np.random.seed(int(seed & 0xFFFFFFFF))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Optional noise schedule
    def get_instance_noise(epoch, max_noise=0.1, max_decay_epochs=10000):
        return max_noise * max(0.0, 1.0 - epoch / max_decay_epochs)

    # Script directory (for params/ckpt); unaffected by CWD changes
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()  # for notebooks

    results_dir = script_dir / "results_tl_tune_config"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tl_losses.txt").write_text("")  # truncate


    param_file = script_dir / "tl_tune_config.json"
    if not param_file.exists():
        raise FileNotFoundError(f"param file not found: {param_file}")

    checkpoint_file = results_dir / "tl_checkpoint.pth"
    print(f"checkpoint file: {checkpoint_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Load params
    with open(param_file, "r") as f:
        params = json.load(f)

    latent_dim    = params["latent_dim"]
    base_channels = params["base_channels"]
    batch_size    = params["batch_size"]
    output_shape  = tuple(params["output_shape"])

    print(f"Batch size: {batch_size}")

    # Load generator
    from GAN_model.gan_generator_cnn_v2 import Generator3D as G3D
    checkpoint = torch.load(checkpoint_file, map_location=device)

    generator = G3D(
        latent_dim=latent_dim,
        base_channels=base_channels,
        init_dim=4,
        output_shape=output_shape,
    ).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # Sample
    epoch = 0
    noise_std = get_instance_noise(epoch)

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = generator(z)             # [B, C, D, H, W]
        fake_occupancy = calc_occupancy(fake)
        print(f"calculating fake eoccupancy across all samples in the batch: {fake_occupancy}")

        if noise_std > 0:
            fake = fake + noise_std * torch.randn_like(fake)
        binary = (fake > 0.5).to(torch.uint8)         # threshold once

    # Save voxels to CWD (instances/<uuid>/)
    voxel_paths: list[str] = []
    B = binary.shape[0]
    for i in range(B):
        vox = binary[i, 0].cpu().numpy()              # channel 0
        out_name = f"fake_voxel_batch_{i}.npy"
        np.save(out_name, vox)
        voxel_paths.append(out_name)
        
        # saving vtk files
        plot_vtk(binary[i, 0], "./", i, 'fake') 

    print(f"Saved {len(voxel_paths)} voxel files to {Path.cwd()}")

    # Save latents (relative)
    latent_paths = [str(p) for p in save_latents_per_batch(
        Z=z,
        out_dir=".",                     # current dir
        base_name="fake_voxel_batch_",
        ckpt_path=checkpoint_file,
    )]

    print(f"Saved {len(latent_paths)} latent vectors under ./latents/")

    return voxel_paths, latent_paths

