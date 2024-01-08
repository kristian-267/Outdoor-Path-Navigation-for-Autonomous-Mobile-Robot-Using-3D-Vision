import torch.nn as nn

from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = build_model(backbone)

    def forward(self, input_dict):
        seg_logits = self.backbone(input_dict)
        return dict(seg_logits=seg_logits)
