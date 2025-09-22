from pathlib import Path
import random, json
import torch
import numpy as np
import json
import os, secrets
import hashlib, time


def save_latents_npz(Z: torch.Tensor, out_path: str | Path, ids: list[str], **arrays):
    """ arrays can be numpy arrays aligned with N (e.g., pdrop, psi, labels)"""
    out_path = Path(out_path); #out_path = out_path.parent.mkdir(parents=True, exist_ok=True)
    Z = Z.detach().to("cpu", dtype=torch.float32).numpy()
    np.savez_compressed(out_path, Z=Z, ids=np.array(ids, dtype=object), **arrays)
    return Z.shape[0]


def save_latents_per_batch(
    Z: torch.Tensor,                 # [B, latent_dim]
    out_dir: str | Path,
    base_name: str = "fake_voxel_batch_",   # aligns with your voxel filenames
    ckpt_path: str | Path | None = None,
) -> list[Path]:
    """
    Saves each z_i as <base_name>{i}.latent.npy in out_dir/latents/
    Also writes a manifest (latents_manifest.json) with index->path mapping.
    Returns list of saved paths ordered by batch index.
    """
    out_dir = Path(out_dir)
    lat_dir = out_dir #/ "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)

    Z = Z.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    B = Z.shape[0]

    def sha256sum(p: Path) -> str | None:
        if not p: return None
        p = Path(p)
        if not p.exists(): return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1<<20), b""):
                h.update(b)
        return h.hexdigest()

    ckpt_hash = sha256sum(Path(ckpt_path)) if ckpt_path else None

    paths = []
    for i in range(B):
        p = lat_dir / f"{base_name}{i}.latent.npy"
        np.save(p, Z[i], allow_pickle=False)
        paths.append(p)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latent_dim": int(Z.shape[1]),
        "count": B,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_sha256": ckpt_hash,
        "items": [
            {"index": i, "latent_path": str(paths[i])}
            for i in range(B)
        ],
        "note": "Latents saved by index; rename/bind to exp_uuid after server returns uuids."
    }
    with (out_dir / "latents_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return paths





def voxel_generator(output_dir: str, seed: int | None = None, deterministic: bool = False):
    """
    Generate and save a batch of voxels from a pre-trained GAN generator.
    Saves fake_voxel_batch_{i}.npy in <output_dir>/results_tl_tune_config
    """

    # ------------------ Device & Repro ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # seed = 42
    # torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     # torch.use_deterministic_algorithms(True)  # uncomment for strict determinism

    
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
        torch.backends.cudnn.benchmark = True  # faster; allows minor algo variability

    # ------------------ Noise decay ------------------
    def get_instance_noise(epoch, max_noise=0.1, max_decay_epochs=10000):
        return max_noise * max(0.0, 1.0 - epoch / max_decay_epochs)

    # ------------------ Paths ------------------
    
    output_dir = Path(output_dir).resolve()

    try:
        current_dir = Path(__file__).resolve().parent
    except NameError:
        current_dir = Path.cwd()  # fallback for notebooks
    
    #print(current_dir)
    dirname = current_dir / "results_tl_tune_config"
    dirname.mkdir(parents=True, exist_ok=True)

    (dirname / "tl_losses.txt").write_text("")  # truncate


    param_file = current_dir / "tl_tune_config.json"
    if not param_file.exists():
        raise FileNotFoundError("param file not found: {param_file}")

    checkpoint_file = dirname / "tl_checkpoint.pth"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # ------------------ Hyperparams ------------------
    # If you actually need hyper-params from file, use this and remove `params` or merge dicts.
    # param_dict = json.loads((current_dir / "hyper-params.json").read_text())

    with open(param_file, 'r') as f:
        params = json.load(f)

    latent_dim     = params["latent_dim"]
    base_channels  = params["base_channels"]
    batch_size     = params["batch_size"]
    output_shape   = tuple(params["output_shape"])
    lr_g           = params["lr_g"]
    betas          = tuple(params["betas"])

    # ------------------ Load Generator ------------------
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

    # ------------------ Sample ------------------
    epoch = 0
    noise_std = get_instance_noise(epoch)
    z = torch.empty(batch_size, latent_dim, device=device)
    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = generator(z)                     # [B, C, D, H, W]
        fake = fake + noise_std * torch.randn_like(fake)
        binary = (fake > 0.5).to(torch.uint8)   # threshold once

    #print(binary.shape)
    # ------------------ Save ------------------
    latent_saved = []
    saved = []
    B = binary.shape[0]
    for i in range(B):
        vox = binary[i, 0].cpu().numpy()  # use channel 0
        # out_path = output_dir / f"fake_voxel_batch_{i}.npy"
        out_path = f"fake_voxel_batch_{i}.npy"
        np.save(out_path, vox)
        saved.append(str(out_path))

    print(f"Saved {len(saved)} voxel files to {dirname}")

    # Save latent vectors
    #lat_lens = save_latents_npz(z, out_path="./latents.npz")


    # After you generated B voxels with z_batch (shape [B, latent_dim]):
    runs_dir = Path(current_dir) / "instances" / instance_uuid
    
    latent_paths = save_latents_per_batch(
        Z=z, 
        out_dir=runs_dir, 
        base_name="fake_voxel_batch_",
        ckpt_path= checkpoint_file,
    )



    print(f"Saved {lat_lens} latent vectors to {dirname}")

    return saved

