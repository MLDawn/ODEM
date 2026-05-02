import torch
def convert(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    else:
        return obj