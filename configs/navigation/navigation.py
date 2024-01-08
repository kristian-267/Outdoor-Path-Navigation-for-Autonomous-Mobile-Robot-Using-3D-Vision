path_cfg = dict(
    model_path = '/mnt/Documents/DTU/Thesis/thesis/model/model_state_dict_best.pth',
    bag_path = '/mnt/Documents/DTU/Thesis/data/raw/20230315_151854.bag'
)
model_cfg = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SPVCNN",
        in_channels=6,
        out_channels=4,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 2, 2, 2, 2, 2, 2, 2)
    )
)
voxelize_cfg = dict(
    voxel_size=0.05,
    mode='train',
    keys=("coord", "color"),
    return_discrete_coord=True
)
camera_cfg = dict(
    resolution=(640, 480),
    num_frame=30
)
grid_cfg = dict(
    grid_size=0.3
)
