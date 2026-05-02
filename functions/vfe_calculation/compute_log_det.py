import torch

def compute(C, verbose=False, jitter_eps=1e-6, max_tries=10):
    """
    Attempts to compute log(det(C)) using spm_logdet.
    Adds jitter iteratively if NaN is returned.
    Returns NaN if all attempts fail.
    """
    for i in range(max_tries):
        logdet = log_det(C)
        if not torch.isnan(logdet) and not torch.isinf(logdet):
            return logdet

        # Add jitter: scaled identity matrix
        C = C + jitter_eps * torch.eye(C.size(0), device=C.device)
        jitter_eps *= 10  # progressively larger if needed

        if verbose:
            print(f"[safe_logdet] Attempt {i+1}: Jitter added (eps={jitter_eps:.1e})")

    # If we exhausted all tries
    if verbose:
        print("[safe_logdet] Failed to compute logdet after jittering. Returning NaN.")
    return torch.tensor(float('nan'), device=C.device)

def log_det(C: torch.Tensor,
                  rtol: float = 1e-12,
                  atol: float = 1e-15,
                  svd_floor: float = 0.0) -> torch.Tensor:
    """
    Stable log(det(C)) for covariance/precision-like matrices,
    similar to SPM's spm_logdet.m. Always returns a scalar tensor.

    - Tries Cholesky if symmetric positive-definite
    - Falls back to SVD-based pseudo-logdet if rank-deficient
    - Returns log|det(C)| as a torch scalar (ignores sign)
    """
    if C.numel() == 0 or C.shape[-1] != C.shape[-2]:
        return torch.tensor(float("nan"), dtype=C.dtype, device=C.device)

    # Symmetry check
    if torch.allclose(C, C.T, rtol=rtol, atol=atol):
        # SPD attempt
        jitter = 0.0
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(
                    C if jitter == 0.0 else C + jitter * torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
                )
                return 2.0 * torch.log(torch.diagonal(L)).sum()
            except RuntimeError:
                jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
        # Fall through to SVD pseudo-logdet if Cholesky fails

    # General or fallback path: SVD pseudo-logdet
    s = torch.linalg.svdvals(C)
    if svd_floor > 0.0:
        floor = svd_floor * s.max()
        s = torch.clamp(s, min=floor)
    else:
        s = s[s > max(atol, rtol * s.max())]
    if s.numel() == 0:
        return torch.tensor(float("-inf"), dtype=C.dtype, device=C.device)
    return torch.log(s).sum()


