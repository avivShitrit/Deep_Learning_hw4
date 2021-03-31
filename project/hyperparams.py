def vanilla_gan_hyperparams():
    hypers = dict(
        batch_size=8,
        z_dim=256,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  
            lr=0.0003,
        ),
        generator_optimizer=dict(
            type="Adam",  
            lr=0.0003,
   
        ),
    )
    return hypers

def sn_gan_hyperparams():
    hypers = dict(
        batch_size=8,
        z_dim=256,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type='Adam',
            lr=0.00005,
            betas=(0.4, 0.9)
        ),
        generator_optimizer=dict(
            type='Adam',
            lr=0.00005,
            betas=(0.4, 0.9)
        ),
    )
    return hypers