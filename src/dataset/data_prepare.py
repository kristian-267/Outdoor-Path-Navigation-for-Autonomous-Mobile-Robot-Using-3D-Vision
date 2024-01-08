import copy
import numpy as np
from collections.abc import Mapping, Sequence
import torch
from torch.utils.data.dataloader import default_collate


def data_load(data_path):
    data = torch.load(data_path)
    coord = data["coord"]
    color = data["color"]
    data = dict(coord=coord, color=color)
    return data

def normalize_color(data_dict):
    if "color" in data_dict.keys():
        data_dict["color"] = data_dict["color"] / 127.5 - 1
    return data_dict

def center_shift(data_dict, apply_z=False):
    old_data_dict = copy.deepcopy(data_dict)
    if "coord" in data_dict.keys():
        x_min, y_min, z_min = data_dict["coord"].min(axis=0)
        x_max, y_max, _ = data_dict["coord"].max(axis=0)
        if apply_z:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        else:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
        data_dict["coord"] -= shift
    return old_data_dict, data_dict

def to_tensor(data_dict):
    for k in data_dict.keys():
        if isinstance(data_dict[k], np.ndarray) and np.issubdtype(data_dict[k].dtype, int):
            data_dict[k] = torch.from_numpy(data_dict[k]).long()
        elif isinstance(data_dict[k], np.ndarray) and np.issubdtype(data_dict[k].dtype, np.floating):
            data_dict[k] = torch.from_numpy(data_dict[k]).float()
    data_dict["feat"] = torch.cat([data_dict[key].float() for key in ("coord", "color")], dim=1)
    data_dict["offset"] = torch.tensor([data_dict["coord"].shape[0]])
    return data_dict

def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)
