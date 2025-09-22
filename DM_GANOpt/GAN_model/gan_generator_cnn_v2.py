# -*- coding: utf-8 -*-
"""
Created on Fri May 07 10:00:00 2025

@author: hazizi
"""

# import torch
# import torch.nn as nn
# import torch.nn.functional
# import torch.optim
# import numpy as np
# import matplotlib.pyplot as plt

        
#  # CNN-based structure generator
# class Generator3D(nn.Module):
#     def __init__(self, latent_dim=100, base_channels=64, output_shape=(32, 32, 32), sigmoid_scale=3.0):
#         super().__init__()

#         self.latent_dim = latent_dim
#         self.base_channels = base_channels
#         self.init_dim = 4
#         self.fc = nn.Linear(self.latent_dim, self.base_channels * self.init_dim**3)

#         self.sigmoid_scale = sigmoid_scale  # new

#         # kernel_size, padding and stride are chosen to double the size of the input volume
#         self.deconv_blocks = nn.Sequential(
#             self._deconv_block(base_channels, base_channels//2),
#             self._deconv_block(base_channels//2, base_channels//4),
#             nn.ConvTranspose3d(base_channels//4, 1, kernel_size=4, stride=2, padding=1),
#             # nn.Sigmoid() 
#         ) 


#     def _deconv_block(self, in_channels, out_channels):
#         return  nn.Sequential(
#                     nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
#                     nn.BatchNorm3d(out_channels),
#                     nn.ReLU(inplace=True)
#                 )
            
#     def forward(self, latent_vec):
#         batch_size = latent_vec.size(0)
#         x = self.fc(latent_vec)
#         x = x.view(batch_size, self.base_channels, self.init_dim, self.init_dim, self.init_dim)
#         x = self.deconv_blocks(x)
#         x = torch.sigmoid(x * self.sigmoid_scale)
#         return x




import torch
import torch.nn as nn
import math

class Generator3D(nn.Module):
    def __init__(self, latent_dim=100, base_channels=64, output_shape=(32, 32, 32), init_dim=4, sigmoid_scale=3.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.init_dim = init_dim
        self.output_shape = output_shape
        self.sigmoid_scale = sigmoid_scale

        # Linear projection from latent space
        self.fc = nn.Linear(latent_dim, base_channels * init_dim ** 3)

        # Compute number of upsampling layers needed per spatial dimension
        target_dim = output_shape[0]
        num_upsamples = int(math.log2(target_dim // init_dim))
        assert 2 ** num_upsamples * init_dim == target_dim, f"output_shape {output_shape} not compatible with init_dim {init_dim}"

        # Create deconv layers to double the size each time
        deconv_layers = []
        in_ch = base_channels
        for i in range(num_upsamples - 1):
            out_ch = in_ch // 2
            deconv_layers.append(self._deconv_block(in_ch, out_ch))
            in_ch = out_ch

        # Final deconv to output 1 channel
        deconv_layers.append(nn.ConvTranspose3d(in_ch, 1, kernel_size=4, stride=2, padding=1))

        self.deconv_blocks = nn.Sequential(*deconv_layers)

    def _deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, latent_vec):
        batch_size = latent_vec.size(0)
        x = self.fc(latent_vec)
        x = x.view(batch_size, self.base_channels, self.init_dim, self.init_dim, self.init_dim)
        x = self.deconv_blocks(x)
        x = torch.sigmoid(x * self.sigmoid_scale)
        return x








# # select device
# device =torch.device("cuda" if torch.cuda.is_available else "cpu")

# # instantiate the model
# model = Generator3D().to(device)

# # create a latent vector
# batch_size=1
# latent_vec = torch.randn(batch_size, model.latent_dim).to(device)

# # generate structures
# model.eval()
# with torch.no_grad():
#     generated_structure = model(latent_vec)

# # visualize the structures
# binary_voxels = (generated_structure > 0.5).float()

# # save the results
# np.save("voxel_0.npy", binary_voxels[0, 0].cpu().numpy())



# # Load voxel data
# voxels = np.load("voxel_0.npy")  # shape: [32, 32, 32]


# print(voxels)
# # Plot a few slices
# fig, axes = plt.subplots(1, 5, figsize=(15, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(voxels[:, :, i*6], cmap='gray')
#     ax.set_title(f"Slice {i*6}")
#     ax.axis('off')

# plt.tight_layout()
# plt.show()
