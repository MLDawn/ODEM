import torch

# Helper function to move tensors or any data structure to the specified device
def move(obj, device):
    if isinstance(obj, torch.Tensor):
        # If it's a tensor, move it to the device (CPU or GPU)
        return obj.to(device)
    elif isinstance(obj, list):
        # If it's a list, apply the function to each element
        return [move(o, device) for o in obj]
    elif isinstance(obj, tuple):
        # If it's a tuple, apply the function to each element
        return tuple(move(o, device) for o in obj)
    elif isinstance(obj, dict):
        # If it's a dictionary, apply the function to each value
        return {k: move(v, device) for k, v in obj.items()}
    else:
        return obj  # If it's not a tensor or a recognized structure, return it unchanged