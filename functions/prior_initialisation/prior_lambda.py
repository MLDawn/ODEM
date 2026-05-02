from functions import create_multivariate_gaussian, precision_to_lambda

def set(E_pi_x, sigma_lambda_x, E_pi_y, sigma_lambda_y):
    # Convert from precision space to lambda space for precision_eta precision_var
    lambda_eta_x, lambda_var_x = precision_to_lambda.compute(E_pi_x, sigma_lambda_x)
    lambda_eta_y, lambda_var_y = precision_to_lambda.compute(E_pi_y, sigma_lambda_y)

    #  Construct the multi-variate normal priors over the hyper-parameters lambda_x and lambda_y
    p_lambda_x_eta, p_lambda_x_pi = create_multivariate_gaussian.create((1,), lambda_eta_x, lambda_var_x, requires_grad=False)
    p_lambda_y_eta, p_lambda_y_pi = create_multivariate_gaussian.create((1,), lambda_eta_y, lambda_var_y, requires_grad=False)

    return p_lambda_x_eta, p_lambda_x_pi, p_lambda_y_eta, p_lambda_y_pi