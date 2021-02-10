# python3.7
"""Configuration for testing StyleGAN on FF-HQ (256) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

runner_type = 'StyleGANRunner'
gan_type = 'stylegan'
resolution = 256
batch_size = 64

data = dict(
    num_workers=4,
    # val=dict(root_dir='data/ffhq', resolution=resolution),
    val=dict(root_dir='/home-local2/yopog.extra.nobkp/lsun/church_outdoor_val_lmdb', data_format='lmdb',
             resolution=resolution, crop_resize_resolution=256),
)

modules = dict(
    discriminator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        kwargs_val=dict(),
    ),
    generator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        kwargs_val=dict(trunc_psi=0.7, trunc_layers=8, randomize_noise=False),
    )
)
