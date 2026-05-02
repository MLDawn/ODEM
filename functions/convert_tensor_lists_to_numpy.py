import numpy as np
def convert(tensor_lists):
    """
    Converts a list of lists of torch tensors to NumPy arrays.

    Parameters:
    - tensor_lists (list of list of torch.Tensor): Each sublist contains torch tensors.

    Returns:
    - list of np.ndarray: Each entry corresponds to the stacked NumPy array.
    """
    return [np.stack([t.detach().cpu().numpy() for t in tensor_list]) for tensor_list in tensor_lists]
