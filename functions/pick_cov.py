import torch

def pick(cov_new, cov_prev, *, dim, device, dtype, cov_prior=None):
    """Return a usable covariance: prefer cov_new if finite, else cov_prev, else prior, else I*1e2."""
    ok = (cov_new is not None) and torch.isfinite(cov_new).all()
    if ok:
        # re-symmetrize just in case
        return 0.5 * (cov_new + cov_new.T)
    if cov_prev is not None:
        return cov_prev
    if cov_prior is not None:
        return cov_prior
    return torch.eye(dim, device=device, dtype=dtype) * 1e2
