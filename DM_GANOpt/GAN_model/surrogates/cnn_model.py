import torch.nn as nn


# class PhysicsSurrage(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv3d(1, 8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(8, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool3d(1),
#             nn.Flatten(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, 2)
#         )

#     def forward(self, x):
#         return self.net(x)
    


class PhysicsSurrage(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=16, output_dim=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # Outputs shape (B, hidden_dim, 1, 1, 1)
            nn.Flatten(),             # -> (B, hidden_dim)
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.regressor(x)
