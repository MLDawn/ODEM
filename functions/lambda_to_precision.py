import torch
import numpy as np

def compute(lambda_eta, lambda_var):
    """
    Convert the mean and variance of a Gaussian-distributed lambda
    to the mean and variance of the corresponding log-normal precision.

    Supports both PyTorch and NumPy inputs. Returns outputs in the same type.

    Parameters
    ----------
    lambda_eta : torch.Tensor or np.ndarray
        Posterior mean of log-precision (lambda), shape (d,)
    lambda_var : torch.Tensor or np.ndarray
        Posterior variance of log-precision, shape (d,)

    Returns
    -------
    precision_eta : torch.Tensor or np.ndarray
        Expected value of precision = E[exp(lambda)], shape (d,)
    precision_var : torch.Tensor or np.ndarray
        Variance of precision = Var[exp(lambda)], shape (d,)
    """
    input_was_numpy = False

    if isinstance(lambda_eta, np.ndarray) and isinstance(lambda_var, np.ndarray):
        lambda_eta = torch.from_numpy(lambda_eta)
        lambda_var = torch.from_numpy(lambda_var)
        input_was_numpy = True
    elif not isinstance(lambda_eta, torch.Tensor) or not isinstance(lambda_var, torch.Tensor):
        raise TypeError("Inputs must both be either torch.Tensor or numpy.ndarray")

    exp_term = torch.exp(lambda_var)
    precision_eta = torch.exp(lambda_eta + 0.5 * lambda_var)
    precision_var = (exp_term - 1.0) * torch.exp(2 * lambda_eta + lambda_var)

    if input_was_numpy:
        return precision_eta.numpy(), precision_var.numpy()
    else:
        return precision_eta, precision_var
