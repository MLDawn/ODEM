from functions import create_multivariate_gaussian

def set(theta_eta, theta_sigma, theta_shape):
    # Defining priors over the parameters
    p_theta_eta, p_theta_pi = create_multivariate_gaussian.create(theta_shape, theta_eta, theta_sigma, requires_grad=False)
    return p_theta_eta, p_theta_pi