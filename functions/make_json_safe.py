import torch
def convert(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    else:
        return str(obj)  # fallback for unexpected types
