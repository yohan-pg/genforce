# python3.7
"""Configuration for training StyleGAN on LSUN cars dataset containing 5 520 753 images.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

runner_type = 'StyleGANRunner'
gan_type = 'stylegan'
resolution = 256
batch_size = 4
val_batch_size = 64
total_img = 5_000_000

# Training dataset is repeated at the beginning to avoid loading dataset
# repeatedly at the end of each epoch. This can save some I/O time.
data = dict(
    num_workers=4,
    repeat=20,
    # train=dict(root_dir='data/ffhq', resolution=resolution, mirror=0.5),
    # val=dict(root_dir='data/ffhq', resolution=resolution),
    train=dict(root_dir='data/lsun/car', data_format='lmdb',
               resolution=resolution, mirror=0.5, crop_resize_resolution=256),
    val=dict(root_dir='data/lsun/car', data_format='lmdb',
             resolution=resolution, crop_resize_resolution=256),
)

controllers = dict(
    RunningLogger=dict(every_n_iters=10),
    ProgressScheduler=dict(
        every_n_iters=1, init_res=8, minibatch_repeats=4,
        lod_training_img=600_000, lod_transition_img=600_000,
        batch_size_schedule=dict(res4=64, res8=32, res16=16, res32=8),
    ),
    Snapshoter=dict(every_n_iters=50000, first_iter=True, num=200),
    FIDEvaluator=dict(every_n_iters=50000, first_iter=True, num=50000),
    Checkpointer=dict(every_n_iters=100000, first_iter=True),
)

modules = dict(
    discriminator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    generator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(w_moving_decay=0.995, style_mixing_prob=0.9,
                          trunc_psi=1.0, trunc_layers=0, randomize_noise=True),
        kwargs_val=dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False),
        g_smooth_img=10_000,
    )
)

loss = dict(
    type='LogisticGANLoss',
    d_loss_kwargs=dict(r1_gamma=10.0),
    g_loss_kwargs=dict(),
)
