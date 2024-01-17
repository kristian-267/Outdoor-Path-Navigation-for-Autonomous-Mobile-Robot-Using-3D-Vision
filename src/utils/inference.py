import torch
import torch.nn.functional as F

from src.dataset.data_prepare import normalize_color, center_shift, to_tensor

def inference(data, model):
    model.eval()
    data = data_process(data)
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cpu()
    with torch.no_grad():
        output = model(data)
        pred = output["seg_logits"]
        pred = F.softmax(pred, -1)
    return pred.max(1)[1].data.numpy()

def data_process(data):
    data = normalize_color(data)
    _, data = center_shift(data, apply_z=False)
    data = to_tensor(data)
    return data
