def run_training(dic: dict, data: list[dict], seed: int | None = None, deterministic: bool = False):
    # ------------------ imports ------------------
    import os, random, secrets
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    from gan_generator_cnn_v2 import Generator3D as g3d
    from models.structure_assessor import Assessor
    from functions.utils import calc_occupancy

    best_param_dict = dic

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
    current_dir = os.path.abspath(os.path.dirname(__file__))
    out_dir = os.path.join(current_dir, "results_tl_tune_config")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_file = os.path.join(out_dir, "tl_checkpoint.pth")
    loss_fname = os.path.join(out_dir, "tl_losses.txt")

    # ------------------ hparams ------------------
    latent_dim    = int(best_param_dict["latent_dim"])
    base_channels = int(best_param_dict["base_channels"])
    batch_size    = int(best_param_dict["batch_size"])
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
    recon_weight          = float(best_param_dict.get("recon_weight", 0.0))  # optional L1-to-targets term

    D, H, W = output_shape

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

    # ------------------ load targets & physics ------------------
    # data = [{"file": "/path/to/voxel.npy", "pdrop": 138.49}, ...]
    file_paths = [x.get("file") for x in data]
    if any(fp is None for fp in file_paths):
        raise ValueError("Every item in `data` must have a 'file' key")

    arrays = []
    for f in file_paths:
        arr = np.load(f, mmap_mode="r")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1 and arr.size == D * H * W:
            arr = arr.reshape(D, H, W)
        if tuple(arr.shape) != (D, H, W):
            raise ValueError(f"{f}: expected shape {(D,H,W)}, got {arr.shape}")
        arrays.append(arr)

    B = len(arrays)
    # enforce batch size if needed (optional)
    if B != batch_size:
        # either adapt, or warn; here we adapt to the data batch
        batch_size = B

    targets = torch.empty((B, 1, D, H, W), dtype=torch.float32, device=device)
    for i, arr in enumerate(arrays):
        targets[i, 0] = torch.from_numpy(arr).to(device)

    # NOTE: physics list is constant w.r.t. generator; no gradients.
    phy_params = torch.tensor(
        [float(x.get("pdrop", 0.0)) for x in data],
        dtype=torch.float32, device=device
    )

    # ------------------ training ------------------
    loss_history = []
    for epoch in range(epochs):
        generator.train()
        optimizer_g.zero_grad()

        # Build latent codes. Adjust shape to your g3d input expectation.
        # If g3d expects [B, latent_dim]:
        z = torch.randn(batch_size, latent_dim, device=device)
        # If g3d expects 5D, use: z = torch.randn(batch_size, latent_dim, 1, 1, 1, device=device)

        fake = generator(z)  # expect [B,1,D,H,W]
        if fake.shape != (batch_size, 1, D, H, W):
            raise RuntimeError(f"Generator output {fake.shape} != {(batch_size,1,D,H,W)}")

        # Assessor-based terms on *generated* data
        fake_occupancy = calc_occupancy(fake)
        scores = assessor(
            fake,
            target_occupancy=target_occupancy,
            occupancy_weight=max_occ_weight,
            resintrap_weight=max_resintrap_weight,
        )
        overhang_score = scores["overhang"]           # scalar tensor
        occupancy_score = scores["occupancy"]         # scalar tensor
        void_mask, resintrap_score = scores["resintrap"]  # (B,1,D,H,W), scalar

        # Optional clamps and shaping
        resintrap_score = torch.clamp(resintrap_score, max=10.0)
        voxelwise_penalty = resintrap_vox_w * (fake * void_mask).mean()

        overhang_score  = overhang_score  * (1 + 1.5 * torch.sigmoid(100 * (overhang_score - 0.01)))
        occ_diff        = (fake_occupancy - target_occupancy).abs()
        occupancy_score = occupancy_score * (1 + 10 * torch.sigmoid(30 * (occ_diff - 0.05)))

        # (Optional) reconstruction: pull fake toward provided targets
        recon_loss = F.l1_loss(fake, targets)

        # “Physics” term (constant; no gradients unless you add a differentiable surrogate)
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
            + recon_weight * recon_loss
            + dynamic_alpha_cc * physics_loss  # NOTE: const wrt G unless surrogate is used
        )

        loss.backward()
        optimizer_g.step()
        loss_history.append(float(loss.detach().cpu()))

        print(
            f"[Epoch {epoch:04d}] "
            f"loss={loss.item():.4f}  overhang={overhang_score.item():.4f}  "
            f"occupancy={occupancy_score.item():.4f}  resintrap={resintrap_score.item():.4f}  "
            f"recon={recon_loss.item():.4f}  phys={physics_loss.item():.4f}  "
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

