_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 32  # bs: total bs in all gpus
mix_prob = 0.5
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SPVCNN",
        in_channels=6,
        out_channels=4,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 2, 2, 2, 2, 2, 2, 2)
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             ignore_index=-1)
    ]
)

# scheduler settings
epoch = 300
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.005)
scheduler = dict(type="MultiStepLR", milestones=[0.8, 0.9], gamma=1.0)

# dataset settings
dataset_type = "DTU_TrailDataset"
data_root = "/work3/s212661/data/dtu_trail" # Where the dataset stored

data = dict(
    num_classes=4,
    ignore_index=-1,
    names=["others", "trail", "pavement", "motorway"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Voxelize", voxel_size=0.05, hash_type="fnv", mode="train",
                 keys=("coord", "color", "segment"), return_discrete_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "segment"), feat_keys=["coord", "color"])
        ],
        cache=True,
        test_mode=False
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Voxelize", voxel_size=0.05, hash_type="fnv", mode="train",
                 keys=("coord", "color", "segment"), return_discrete_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "discrete_coord", "segment"), feat_keys=["coord", "color"])
        ],
        test_mode=False),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor")
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="Voxelize",
                voxel_size=0.05,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_discrete_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "discrete_coord", "index"),
                    feat_keys=("coord", "color"))
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [dict(type="RandomScale", scale=[0.9, 0.9]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[0.95, 0.95]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1, 1]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.05, 1.05]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.1, 1.1]),
                 dict(type="RandomFlip", p=1)],
            ]
        )
    )
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="DataCacheOperator", data_root=data_root, split=data["train"]["split"])
]
