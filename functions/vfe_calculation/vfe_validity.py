import torch
def check(vfe, step_info=""):
    if torch.isnan(vfe) or torch.isinf(vfe):
        raise ValueError(f"Invalid VFE encountered (NaN or Inf) at step: {step_info}")
