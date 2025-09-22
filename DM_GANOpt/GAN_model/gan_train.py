def run_training(data: list[dict], seed: int | None = None, deterministic: bool = False):
    """
    Expects `data` like:
    [
      {
        "exp_uuid": "...",
        "voxel_tar": "/abs/path/<uuid>_fake_voxel_batch_0.npy.tar.gz",  # or a plain .npy path
        "latent":    "/abs/path/latents/<uuid>_fake_voxel_batch_0.latent.npy",
        "pdrop": 138.49007
      },
      ...
    ]
    """

    # ------------------ imports ------------------
    import os, io, re, json, random, secrets, tarfile
    from pathlib import Path
    from typing import List
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.optim as optim


    # Load generator
    from GAN_model.gan_generator_cnn_v2 import Generator3D as g3d

    from GAN_model.models.structure_assessor import Assessor
    from GAN_model.functions.utils import calc_occupancy

    #best_param_dict = dic

    # ------------------ device & reproducibility ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is None:
        seed = secrets.randbits(64)
    torch.manual_seed(seed)
    np.random.seed(int(seed & 0xFFFFFFFF))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # ------------------ paths ------------------
    script_dir = os.path.abspath(os.path.dirname(__file__))
    out_dir = os.path.join(script_dir, "results_tl_tune_config")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_file = os.path.join(out_dir, "tl_checkpoint.pth")
    loss_fname = os.path.join(out_dir, "tl_losses.txt")


    param_file = os.path.join(script_dir, "tl_tune_config.json")
   
    try:
        with open(param_file, "r") as f:
            best_param_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"param file not found: {param_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {param_file}: {e}") from e


    # ------------------ hparams ------------------
    latent_dim    = int(best_param_dict["latent_dim"])
    base_channels = int(best_param_dict["base_channels"])
    batch_size    = int(best_param_dict["batch_size"])  # will be overridden by len(data)
    epochs        = int(best_param_dict.get("epochs", 1))
    lr_g          = float(best_param_dict["lr_g"])
    betas         = tuple(best_param_dict["betas"])
    output_shape  = tuple(best_param_dict["output_shape"])  # (D,H,W)

    alpha_cc              = float(best_param_dict["alpha_cc"])
    target_occupancy      = float(best_param_dict["target_occupancy"])
    max_resintrap_weight  = float(best_param_dict["resintrap_weight"])
    max_occ_weight        = float(best_param_dict["occupancy_weight"])
    max_alpha_ove         = float(best_param_dict["alpha_ove"])
    warmup_epochs         = int(best_param_dict.get("warmup_epochs", 0))
    warmup_start_epoch    = int(best_param_dict.get("warmup_start_epoch", 0))
    save_interval         = int(best_param_dict.get("save_interval", 0))
    resintrap_vox_w       = float(best_param_dict.get("resintrap_voxelwise_loss_weight", 0.0))
    recon_weight          = float(best_param_dict.get("recon_weight", 0.0))  # optional L1-to-targets

    D, H, W = output_shape

    # ------------------ helpers ------------------
    def _load_npy_maybe_tar(path: str | Path) -> np.ndarray:
        """Load a .npy directly; if path is a .tar.* file, read the first .npy member from it."""
        p = Path(path)
        if p.suffix == ".npy":
            return np.load(p, mmap_mode="r").astype(np.float32, copy=False)
        # read from tar.* (gz/bz2/xz)
        with tarfile.open(p, "r:*") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith(".npy")), None)
            if member is None:
                raise FileNotFoundError(f"No .npy inside tar: {p}")
            with tar.extractfile(member) as fh:
                data = fh.read()
            return np.load(io.BytesIO(data)).astype(np.float32, copy=False)

    # ------------------ load latents, targets & physics ------------------
    if not data or not isinstance(data, list):
        raise ValueError("`data` must be a non-empty list of dicts with keys: 'latent', 'voxel_tar', 'pdrop'.")

    latents_list: List[np.ndarray] = []
    arrays: List[np.ndarray] = []
    phy_list: List[float] = []

    for i, item in enumerate(data):
        lat_path = item.get("latent")
        vox_path = item.get("voxel_tar") or item.get("file")  # allow fallback if someone uses 'file'
        if not lat_path:
            raise ValueError(f"data[{i}] missing 'latent' path")
        if not vox_path:
            raise ValueError(f"data[{i}] missing 'voxel_tar' (or 'file') path")

        # latent
        z_i = np.load(lat_path).astype(np.float32, copy=False)
        # print(f"z_i.dim: {z_i.ndim}")
        if z_i.ndim != 1:
            print("latent_vec before reshape: ", z_i.shape)
            z_i = z_i.reshape(-1)
            print("latent_vec after reshape: ", z_i.shape)
        latents_list.append(z_i)

        # voxel
        arr = _load_npy_maybe_tar(vox_path)
        # print(f"voxel_dim: {arr.ndim}, voxel_size: {arr.size}")
        if arr.ndim == 1 and arr.size == D * H * W:
            arr = arr.reshape(D, H, W)
        if tuple(arr.shape) != (D, H, W):
            raise ValueError(f"{vox_path}: expected voxel shape {(D,H,W)}, got {arr.shape}")
        arrays.append(arr)

        # physics
        pdrop = float(item.get("pdrop", 0.0))
        phy_list.append(pdrop)

    # stack tensors
    Z = torch.from_numpy(np.stack(latents_list, axis=0)).to(device)  # [B, latent_dim]
    if Z.ndim != 2 or Z.shape[1] != latent_dim:
        raise ValueError(f"Loaded latents have shape {tuple(Z.shape)}; expected [B,{latent_dim}]")

    targets = torch.stack([torch.from_numpy(a) for a in arrays], dim=0).to(device)  # [B,D,H,W]
    targets = targets.unsqueeze(1).contiguous()  # [B,1,D,H,W]

    phy_params = torch.tensor(phy_list, dtype=torch.float32, device=device)

    B = len(arrays)
    batch_size = B  # adapt to provided data

    # ------------------ generator ------------------
    generator = g3d(
        latent_dim=latent_dim,
        base_channels=base_channels,
        init_dim=4,
        output_shape=output_shape,
    ).to(device)

    if os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=device)
        sd = ckpt.get("generator_state_dict")
        if sd:
            generator.load_state_dict(sd, strict=False)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)

    # ------------------ assessor ------------------
    assessor = Assessor(
        occupancy_weight=max_occ_weight,
        resintrap_weight=max_resintrap_weight
    ).to(device)

    # ------------------ training ------------------
    loss_history = []
    for epoch in range(epochs):
        generator.train()
        optimizer_g.zero_grad()

        # Use provided latents
        z = Z  # [B, latent_dim]
        fake = generator(z)  # expect [B,1,D,H,W]
        if fake.shape != (batch_size, 1, D, H, W):
            raise RuntimeError(f"Generator output {fake.shape} != {(batch_size,1,D,H,W)}")

        # Assessor-based terms
        fake_occupancy = calc_occupancy(fake)

        print(f"Calculated fake occupancy for all samples in the batch: {fake_occupancy}")
        scores = assessor(
            fake,
            target_occupancy=target_occupancy,
            occupancy_weight=max_occ_weight,
            resintrap_weight=max_resintrap_weight,
        )
        overhang_score = scores["overhang"]               # scalar
        occupancy_score = scores["occupancy"]             # scalar
        void_mask, resintrap_score = scores["resintrap"]  # (B,1,D,H,W), scalar

        # Optional shaping
        resintrap_score = torch.clamp(resintrap_score, max=10.0)
        voxelwise_penalty = resintrap_vox_w * (fake * void_mask).mean()

        overhang_score  = overhang_score  * (1 + 1.5 * torch.sigmoid(100 * (overhang_score - 0.01)))
        occ_diff        = (fake_occupancy - target_occupancy).abs()
        occupancy_score = occupancy_score * (1 + 10 * torch.sigmoid(30 * (occ_diff - 0.05)))

        # Reconstruction toward target voxels
        #recon_loss = F.l1_loss(fake, targets)

        # Physics term (constant wrt G unless you plug in a differentiable surrogate)
        physics_loss = phy_params.mean()

        # Warmup progression for alpha_cc
        if warmup_epochs > 0:
            progress = min(1.0, max(0.0, (epoch - warmup_start_epoch) / warmup_epochs))
        else:
            progress = 1.0
        dynamic_alpha_cc = progress * alpha_cc

        # Total loss
        loss = (
            max_alpha_ove * overhang_score
            + occupancy_score
            + resintrap_score
            + voxelwise_penalty
            + 1e-3 * (fake ** 2).mean()    # small L2
            #+ recon_weight * recon_loss
            + dynamic_alpha_cc * physics_loss
        )

        loss.backward()
        optimizer_g.step()
        loss_history.append(float(loss.detach().cpu()))

        print(
            f"[Epoch {epoch:04d}] "
            f"loss={loss.item():.4f}  overhang={overhang_score.item():.4f}  "
            f"occupancy={occupancy_score.item():.4f}  resintrap={resintrap_score.item():.4f}  "
            #f"recon={recon_loss.item():.4f}  phys={physics_loss.item():.4f}  "
            f"alpha_cc*={dynamic_alpha_cc:.3f}"
        )

        if save_interval and (epoch + 1) % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "loss_g_list": loss_history,
                },
                checkpoint_file,
            )

    # final save
    torch.save(
        {
            "epoch": epochs - 1,
            "generator_state_dict": generator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "loss_g_list": loss_history,
        },
        checkpoint_file,
    )

    # write losses once
    with open(loss_fname, "w") as f:
        for v in loss_history:
            f.write(f"{v}\n")

