input_dict = {
    '0': {
        'module_name': 'structure_generation',
        'inputs': [
            {
                'keys': [
                    "latent_dim",
                    "base_channels",
                    "sigmoid_scale",
                    "lr_g",
                    "lr_d",
                    "epochs",
                    "betas",
                    "batch_size",
                    "alpha_ove",
                    "resintrap_weight",
                    "occupancy_weight",
                    "max_surface_penalty",
                    "resintrap_voxelwise_loss_weight",
                    "alpha_cc",
                    "warmup_epochs",
                    "save_interval"
                ],
                'values': [
                    100,
                    64,
                    3,
                    0.0001,
                    0.00005,
                    5000,
                    [0.5, 0.999],
                    16,
                    50.0,
                    100.0,
                    100.0,
                    10,
                    25,
                    1.0,
                    5000,
                    5000
                ]
            }
        ]
    },
}
