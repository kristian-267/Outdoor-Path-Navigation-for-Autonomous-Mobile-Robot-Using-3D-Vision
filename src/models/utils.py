"""
General Utils for Models
"""

import torch

def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().to(offset.device)
