import torch
import collections

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.models import build_model

path_cfg = dict(
    model_path = '/mnt/Documents/DTU/Thesis/thesis/model/model_state_dict_best.pth',
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

model_path = path_cfg['model_path']

device = torch.device('cpu')
model = build_model(model_cfg)
model.to(device)

state_dict = torch.load(model_path, map_location=device)
new_state_dict = collections.OrderedDict()
for name, value in state_dict.items():
    if name.startswith("module."):
        name = name[7:]  # module.xxx.xxx -> xxx.xxx
    new_state_dict[name] = value
model.load_state_dict(new_state_dict, strict=True)

print('end')
