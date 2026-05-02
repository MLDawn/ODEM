import torch
from functions.vfe_calculation import compute_autocov

def compute(q_mu_lambda, num_generalised_coordinates, d, h_value, lambda_value):
    """
    Constructs the generalised precision matrix over generalised coordinates.

    Args:
        q_mu_lambda (torch.Tensor): Scalar log-precision (lambda)
        num_generalised_coordinates (int): Number of generalised orders
        d (int): State dimensionality
        h_value (torch.Tensor): Time step (requires_grad=True)
        lambda_value (float or torch.Tensor): Temporal decay parameter

    Returns:
        gen_pi (torch.Tensor): Generalised precision matrix of shape [(k*d), (k*d)]
    """

    # --------------------------------------------------
    # 1) State-space precision (isotropic)
    # --------------------------------------------------
    pi_scalar = torch.exp(q_mu_lambda)
    pi = pi_scalar * torch.eye(d, dtype=q_mu_lambda.dtype, device=q_mu_lambda.device)

    # --------------------------------------------------
    # 2) Autocovariance over generalised coordinates
    # --------------------------------------------------
    S = compute_autocov.compute(num_generalised_coordinates, h_value, lambda_value)

    # Enforce symmetry (covariance must be symmetric)
    S = 0.5 * (S + S.T)

    # --------------------------------------------------
    # 3) Minimal, scale-aware jitter for numerical stability
    # --------------------------------------------------
    # Typical variance scale
    scale = S.diagonal().mean()

    # Jitter proportional to scale (no absolute clamping)
    jitter = torch.finfo(S.dtype).eps * scale

    S = S + jitter * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)

    # --------------------------------------------------
    # 4) Stable inversion via Cholesky
    # --------------------------------------------------
    L = torch.linalg.cholesky(S)
    S_inv = torch.cholesky_inverse(L)

    # --------------------------------------------------
    # 5) Generalised precision (Kronecker structure)
    # --------------------------------------------------
    gen_pi = torch.kron(S_inv.contiguous(), pi.contiguous())

    return gen_pi

# def compute(q_mu_lambda, num_generalised_coordinates, d, h_value, lambda_value):
#     """
#     Constructs the generalised precision matrix over generalised coordinates.
#     h_value: Step size (must be requires_grad=True)
#     lambda_value: Temporal decay parameter
#     """
#     # Step 1: Diagonal precision matrix for state dimensions
#     # Scalar precision
#     pi_scalar = torch.exp(q_mu_lambda)
#     # Diagonal precision matrix (same for all dims)
#     pi = pi_scalar * torch.eye(d, dtype=q_mu_lambda.dtype, device=q_mu_lambda.device)
#
#     # Step 2: Compute autocovariance matrix S
#     S = compute_autocov.compute(num_generalised_coordinates, h_value, lambda_value)
#
#     # Optional: symmetrize to avoid tiny numerical asymmetry
#     S = 0.5 * (S + S.T)
#
#     # Step 2.5: Add jitter before inverting S
#     jitter = 1e-9
#     S = S + jitter * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
#
#     # Step 3: Invert S
#     L = torch.linalg.cholesky(S)
#     S_inv = torch.cholesky_inverse(L)
#
#     # Step 4: Kronecker product to construct generalised precision matrix
#     gen_pi = torch.kron(S_inv.contiguous(), pi.contiguous())
#
#
#     return gen_pi
