import numpy as np
def compute(mu: np.ndarray, cov: np.ndarray):
    """
    Approximates E[exp(lambda)] and Cov[exp(lambda)] using the Delta method.
    mu:  [d] mean vector of lambda
    cov: [d, d] covariance of lambda
    Returns:
        exp_mu: [d] approximate expectation of exp(lambda)
        cov_pi: [d, d] approximate covariance of exp(lambda)
    """
    exp_mu = np.exp(mu)                  # [d]
    J = np.diag(exp_mu)                  # Jacobian of exp at mu, shape [d, d]
    cov_pi = J @ cov @ J.T               # Delta method approximation
    return exp_mu, cov_pi

# def compute(mu: np.ndarray, cov: np.ndarray):
#     """
#     Exact E[exp(lambda)] and Cov[exp(lambda)] when lambda ~ N(mu, cov).
#     mu:  [d]
#     cov: [d, d]
#     """
#     var = np.diag(cov)                              # [d]
#     exp_mu = np.exp(mu + 0.5 * var)                # exact mean of log-normal
#
#     d = mu.shape[0]
#     cov_pi = np.zeros_like(cov)
#     for i in range(d):
#         for j in range(d):
#             cov_pi[i, j] = np.exp(
#                 mu[i] + mu[j] + 0.5 * (cov[i, i] + cov[j, j])
#             ) * (np.exp(cov[i, j]) - 1.0)
#
#     return exp_mu, cov_pi
