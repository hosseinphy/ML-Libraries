from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
from .cnn_model import PhysicsSurrage

def train_surrogate(voxel_paths, param_list, batch_size, epochs, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhysicsSurrage().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Prepare dataset
    x_list = []
    y_list = []

    for voxel_file, params in zip(voxel_paths, param_list):
        voxel_np = np.load(voxel_file)
        x_tensor = torch.tensor(voxel_np, dtype=torch.float32).unsqueeze(0)  # shape: (1, D, H, W)
        y_tensor = torch.tensor([params["pressure_drop"], params["vel_sa"]], dtype=torch.float32)
        x_list.append(x_tensor)
        y_list.append(y_tensor)

    x_data = torch.stack(x_list).to(device)  # shape: (N, 1, D, H, W)
    y_data = torch.stack(y_list).to(device)  # shape: (N, 2)

    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
