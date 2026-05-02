import torch
def antisym_A_from_theta(theta: torch.Tensor) -> torch.Tensor:
    """
    Map theta[..., 3] -> A[..., 3, 3], with:
      A[0,1]=a12, A[1,0]=-a12
      A[0,2]=a13, A[2,0]=-a13
      A[1,2]=a23, A[2,1]=-a23
    Diagonal = 0. Supports batching on leading dims.
    """
    assert theta.shape[-1] == 3, "theta must have last dimension 3 (a12, a13, a23)."
    a12, a13, a23 = torch.unbind(theta, dim=-1)

    A = torch.zeros(*theta.shape[:-1], 3, 3, dtype=theta.dtype, device=theta.device)
    A[..., 0, 1] = a12;  A[..., 1, 0] = -a12
    A[..., 0, 2] = a13;  A[..., 2, 0] = -a13
    A[..., 1, 2] = a23;  A[..., 2, 1] = -a23
    return A

def glv(x, theta):
    A = antisym_A_from_theta(theta)
    r = torch.zeros(3, dtype=x.dtype, device=x.device)

    output = x * (r + A @ x)
    return output

def lorenz(x, theta):
    # theta: learnable scalar (rho)
    rho = theta
    sigma = torch.tensor([10.0], dtype=x.dtype, device=x.device)
    beta = torch.tensor([8.0 / 3.0], dtype=x.dtype, device=x.device)
    # calc new values for x
    dx0 = sigma * (x[1] - x[0])
    dx1 = x[0] * (rho - x[2]) - x[1]
    dx2 = x[0] * x[1] - beta * x[2]
    output = torch.stack([dx0, dx1, dx2]).squeeze()
    return output


