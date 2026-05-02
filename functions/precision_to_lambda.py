import torch

def compute(E_pi, sigma_lambda):
    """
    Convert the desired EXPECTATION of a precision variable π
    (which is modeled as lognormal: π = exp(lambda))
    into Gaussian parameters for the corresponding log-precision λ.

    Args:
        E_pi (float or torch.Tensor):
            Desired expectation of the precision (E[π]).
            This is the mean of the lognormal precision distribution.

        sigma_lambda (float):
            Standard deviation of the Gaussian prior over λ (log-precision).

    Returns:
        mu (torch.Tensor):
            Mean of λ such that E[π] = exp(mu + 0.5 * sigma_lambda^2).

        var_lambda (torch.Tensor):
            Variance of λ (sigma_lambda^2).
    """

    # variance in lambda space
    var_lambda = sigma_lambda ** 2

    # convert expectation to log-precision mean
    mu = torch.log(E_pi) - 0.5 * var_lambda

    return mu, var_lambda

