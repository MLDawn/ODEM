import torch
from functools import reduce
import operator
def create(shape , eta_val = torch.tensor(1.0), sigma_val = torch.tensor(1.0), requires_grad=False):
    # if len(shape) >= 2:
    #     mean_dims = reduce(operator.mul, shape)
    # elif len(shape) == 1:
    mean_dims = shape[0]

    if requires_grad:
        mean = torch.rand(mean_dims, requires_grad=requires_grad)
        # Create a covariance matrix (identity matrix for simplicity)
        R = torch.rand(mean_dims,mean_dims)  # Identity matrix as the covariance matrix
        # Step 2: Create a symmetric positive semi-definite matrix by multiplying A with its transpose
        cov_matrix = torch.mm(R, R.t())
    else:
        var_val = sigma_val**2
        mean = eta_val * torch.ones(mean_dims, requires_grad=requires_grad) # eta_val will dictate the vector values
        # Create a covariance matrix (identity matrix for simplicity)
        cov_matrix = var_val * torch.eye(mean_dims) # a diagonal matrix, where the diagonal elements are dictated by var_val

    precision_matrix = torch.inverse(cov_matrix)

    return mean, precision_matrix
