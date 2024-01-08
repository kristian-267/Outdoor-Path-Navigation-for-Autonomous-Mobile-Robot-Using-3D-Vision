import torch
import collections

from src.models import build_model

def load_model(path, cfg):
    device = torch.device('cpu')
    model = build_model(cfg)
    model.to(device)

    state_dict = torch.load(path, map_location=device)
    new_state_dict = collections.OrderedDict()
    for name, value in state_dict.items():
        if name.startswith("module."):
            name = name[7:]  # module.xxx.xxx -> xxx.xxx
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict, strict=True)
    return model
