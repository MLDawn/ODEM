import numpy as np
def compute(prior_theta_eta, prior_theta_pi, q_theta, q_cov_theta):
    # convert posterior covariances into posterior precisions so you can easily apply BPA
    pi_theta = np.linalg.inv(q_cov_theta)  # Shape will be (1001, 4, 4)
    # numbewr of parameter estimates
    N = q_theta.shape[0]
    # the bpa precision
    bpa_pi = np.sum(pi_theta, axis=0) - (N - 1) * prior_theta_pi
    # the bpa expected value
    weighted_sum = np.einsum('nij,ni->j', pi_theta, q_theta) - (N - 1) * (prior_theta_pi @ prior_theta_eta)
    bpa_mu = np.linalg.inv(bpa_pi) @ weighted_sum
    # First convert the BPA precision matrix to covariance
    bpa_cov = np.linalg.inv(bpa_pi)
    return bpa_mu, bpa_cov