import torch
from torch.autograd.functional import jacobian

import torch
from torch.autograd.functional import jacobian

def compute(matrix, variable):
    """
    Computes the Jacobian of the input matrix w.r.t. the variable.

    Args:
        matrix (torch.Tensor): The output tensor (e.g., h_mu) of size 2x2.
        variable (torch.Tensor): The input tensor (e.g., q) of size 2x2.

    Returns:
        torch.Tensor: The Jacobian matrix of size 4x4 (flattened output w.r.t. flattened input).
    """
    # Flatten the matrix and variable for easier computation
    matrix_flat = matrix.view(-1)
    variable_flat = variable.view(-1)

    # Initialize the Jacobian
    jacobian = torch.zeros(matrix_flat.size(0), variable_flat.size(0), requires_grad=False)

    # Temporarily disable gradient tracking to ensure clean behavior
    with torch.no_grad():
        if variable.grad is not None:
            variable.grad.zero_()

    # Compute partial derivatives for each element of the matrix
    for i in range(matrix_flat.size(0)):
        # Compute the gradient of the i-th element of the matrix
        matrix_flat[i].backward(retain_graph=True)

        # Store the gradient in the Jacobian
        jacobian[i, :] = variable.grad.view(-1)

        # Reset gradients to ensure no accumulation
        variable.grad.zero_()

    return jacobian

# This one prodices identical results to the one above, but it uses autograd() and NOT backward()

# def compute(matrix, variable, create_graph=True):
#     """
#     Jacobian d vec(matrix) / d vec(variable)
#     matrix  : same shape as variable
#     variable: requires_grad=True (leaf or non-leaf)
#     returns : (N, N) where N = variable.numel()
#     """
#     # Ensure we can read .grad if 'variable' is non-leaf
#     if variable.grad_fn is not None and not variable.requires_grad:
#         raise ValueError("variable must have requires_grad=True")
#     if variable.grad_fn is not None and getattr(variable, 'retains_grad', None) is None:
#         variable.retain_grad()  # so .grad will be populated if you ever call .backward()
#
#     mat_flat = matrix.reshape(-1)
#     N = variable.numel()
#
#     # Preallocate on same device/dtype
#     J = torch.zeros(mat_flat.size(0), N, device=variable.device, dtype=variable.dtype)
#
#     # Build Jacobian row-by-row with autograd.grad
#     for i in range(mat_flat.size(0)):
#         # gradient of scalar mat_flat[i] w.r.t. variable → same shape as variable
#         gi = torch.autograd.grad(
#             mat_flat[i],
#             variable,
#             retain_graph=True,         # reuse graph for the next rows
#             create_graph=create_graph, # keep graph if you’ll differentiate J later
#             allow_unused=False
#         )[0]
#         J[i, :] = gi.reshape(-1)
#
#     return J


