# ------------------- Imports ----------------------
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from gan_generator_cnn_v2 import Generator3D as g3d
from models.structure_builder import create_voxel_structure as cvs
from models.structure_assessor import Assessor 

import matplotlib.pyplot as plt
import pandas as pd
import os, sys, json, random

from functions.utils import plot_vtk, calc_occupancy, plot_voxel 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from surrogates.cnn_model import PhysicsSurrage
from surrogates.train_model import train_surrogate


# ------------------ Device and Reproducibility ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------- Noise Function for GAN Regularization ------------------
def get_instance_noise(epoch, max_noise=0.1, max_decay_epochs=10000):
    return max_noise * max(0.0, 1.0 - epoch / max_decay_epochs)

# ------------------- EMA Class (not currently used) -------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = type(model)()
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.to(next(model.parameters()).device)
        self.decay = decay
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            ema_params = dict(self.ema_model.named_parameters())
            model_params = dict(model.named_parameters())
            for name in ema_params.keys():
                ema_params[name].data.mul_(self.decay).add_(model_params[name].data, alpha=1 - self.decay)

# ------------------- I/O Setup -------------------
current_dir = os.path.abspath(os.path.dirname(__file__))
phys_param_path = os.path.join(current_dir, "Parameters/physical_params.json")
data_path  = os.path.join(current_dir, "Data")

@print(data_path)
# ------------------- Load Hyperparameters -------------------

#if len(sys.argv) < 2:
#    print("Usage: python ML_model_train.py <config_filename.json>")
#    sys.exit(1)
#
#fname = sys.argv[1]
#if not os.path.exists(fname):
#    print(f"Error: file '{fname}' not found.")
#    sys.exit(1)
#
#with open(fname, 'r') as f:
#    best_param_dict = json.load(f)


#fname = "tl_tune_config.json"
#with open(fname, 'r') as f:
#    best_param_dict = json.load(f)

# ------------------- Result Paths -------------------
dirname = "results_{}".format(fname.strip('.json'))
os.makedirs(dirname, exist_ok=True)
loss_fname = os.path.join(dirname, "tl_losses.txt")
checkpoint_file = os.path.join(dirname, "tl_checkpoint.pth")
open(loss_fname, "w").close()

# ------------------- Unpack Training Params -------------------
latent_dim = best_param_dict['latent_dim']
base_channels = best_param_dict['base_channels']
batch_size = best_param_dict['batch_size']
epochs = best_param_dict['epochs']
lr_g = best_param_dict['lr_g']
betas = tuple(best_param_dict['betas'])
output_shape = tuple(best_param_dict['output_shape'])

alpha_ove = best_param_dict['alpha_ove']
alpha_cc = best_param_dict["alpha_cc"]
target_occupancy = best_param_dict["target_occupancy"]
max_resintrap_weight = best_param_dict["resintrap_weight"]
max_occ_weight = best_param_dict["occupancy_weight"]
max_alpha_ove = best_param_dict["alpha_ove"]
warmup_epochs = best_param_dict["warmup_epochs"]
save_interval = best_param_dict["save_interval"]
resintrap_voxelwise_loss_weight = best_param_dict["resintrap_voxelwise_loss_weight"]

warmup_start_epoch = 0

# ------------------- Generator Setup -------------------
checkpoint = torch.load(checkpoint_file, map_location=device)

# generator = g3d(latent_dim=latent_dim, base_channels=base_channels).to(device)

generator = g3d(
    latent_dim=latent_dim,
    base_channels=base_channels,
    init_dim=4,  # Can also be 8, if compatible
    output_shape=output_shape
).to(device)


generator.load_state_dict(checkpoint['generator_state_dict'])
for param in generator.parameters():
    param.requires_grad = True


optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)

# ------------------- Assessor Module -------------------
assessor = Assessor(input_shape=output_shape, occupancy_weight=10., resintrap_weight=100.).to(device)

# ------------------- Train or Load Surrogate Model -------------------
with open(phys_param_path, 'r') as f:
    phy_params_list = json.load(f)

voxel_paths = [data_path+"/{}".format(f) for f in os.listdir(data_path)]
batch_size = len(phy_params_list)  # override with dataset size


#print(voxel_paths, phy_params_list)

# Train CNN surrogate model to predict [pressure_drop, vel_sa]
physics_model = train_surrogate(voxel_paths, phy_params_list, batch_size=batch_size, epochs=epochs, device=device)
physics_model.eval()
for p in physics_model.parameters():
    p.requires_grad = False

print("passed trainig")
# ------------------- Training Loop -------------------
loss_g_list = []
for epoch in range(epochs):

    print(epoch)
    instance_noise_std = get_instance_noise(epoch)

    z = torch.randn(batch_size, latent_dim).to(device)
    fake_data = generator(z)  # Generated 3D structures

    fake_occupancy = calc_occupancy(fake_data)
    fake_data_noisy_for_g = fake_data + instance_noise_std * torch.randn_like(fake_data)

    progress = min(1.0, max(0.0, (epoch - warmup_start_epoch) / warmup_epochs))
    dynamic_alpha_cc = progress * alpha_cc

    clean_fake = (fake_data_noisy_for_g > 0.5).float()

    scores = assessor(clean_fake, target_occupancy=target_occupancy,
                      occupancy_weight=max_occ_weight, resintrap_weight=max_resintrap_weight)

    overhang_score = scores["overhang"]
    occupancy_score = scores["occupancy"]
    void_mask, resintrap_score = scores["resintrap"]

    resintrap_score = torch.clamp(resintrap_score, max=10.0)
    resintrap_voxelwise_penalty = (fake_data * void_mask).mean()

    # Smooth penalty scaling for overhang and occupancy deviations
    overhang_score *= 1 + 1.5 * torch.sigmoid(100 * (overhang_score - 0.01))
    diff = abs(fake_occupancy - target_occupancy)
    occupancy_score *= 1 + 10 * torch.sigmoid(30 * (diff - 0.05))

    # ------------------- Physics Prediction -------------------
    with torch.no_grad():
        if clean_fake.ndim == 4:
            clean_fake = clean_fake.unsqueeze(1)
        pred_physics = physics_model(clean_fake)
        pred_pres_drop = pred_physics[:, 0]
        pred_vel_sa = pred_physics[:, 1]

    epsilon = 1e-6
    physics_loss = torch.mean(pred_pres_drop / (pred_vel_sa + epsilon))

    # ------------------- Generator Loss -------------------
    loss_g = dynamic_alpha_cc * physics_loss
    loss_g += (max_alpha_ove * overhang_score + occupancy_score + resintrap_score)
    loss_g += 0.001 * torch.mean(fake_data ** 2)  # L2 regularization

    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    loss_g_list.append(loss_g.item())

    # ------------------- Logging and Checkpoints -------------------
    if epoch % 1000 == 0:
        print("Epoch: ", epoch)
        print(f"fake occupancy: {fake_occupancy}\n")

        if epoch % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "loss_g_list": loss_g_list,
            }, checkpoint_file)

        voxel_np = (fake_data[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        np.save(os.path.join(dirname, f"fake_voxel_epoch_{epoch}.npy"), voxel_np)

        print(f"[Debug] Resintrap soft mask stats @ Epoch {epoch}: min={void_mask.min():.4f}, max={void_mask.max():.4f}, mean={void_mask.mean():.4f}")
        print("Dynamic weights:")
        print(f"dynamic CC weight : {dynamic_alpha_cc}")
        print("Scores:")
        print(f"Overhang score: {overhang_score}")
        print(f"Occupancy score: {occupancy_score}")
        print(f"Resintrap score: {resintrap_score}")
        print(f"Carbon capture score (DP/vel_sa): {physics_loss}")

        with open(loss_fname, "w") as f:
            for g in loss_g_list:
                f.write(f"{g}\n")

        plot_vtk(fake_data[0, 0], dirname, epoch, 'fake')

# ------------------- Final Save -------------------
torch.save({
    "epoch": epochs - 1,
    "generator_state_dict": generator.state_dict(),
    "optimizer_g_state_dict": optimizer_g.state_dict(),
    "loss_g_list": loss_g_list,
}, checkpoint_file)

# ------------------- Loss Plot -------------------
plt.figure(figsize=(10, 5))
plt.plot(loss_g_list, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
