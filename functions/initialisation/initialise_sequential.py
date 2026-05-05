from data_utils.params import Params
import torch
from fractions import Fraction


def set(config_name="parameters.yaml"):
    params = Params(config_name)
    params = params.yaml_map
    opt = params['optimizer']
    gm = params['gm']
    device = 'cpu'
    kx, ky = gm['kx'], gm['ky']
    f_name, g_name = gm['dynamics'], gm['likelihood']
    # Generative Process parameters
    gp = params['gp']
    gp_name = gp['name']
    # the time step, used in generating hidden states from the generative process AND in estimating y',y'',...
    dt, T = gp['dt'], gp['T']

    # the name of the algorithm for separation of temporal scales
    algorithm_name = 'ODEM'



    # Observation noise
    y_noise = gm['noise']
    y_wn_mu, y_wn_sigma = y_noise['y_wn_mu'], y_noise['y_wn_sigma']
    y_cn_kernel_size, y_cn_kernel_sigma = y_noise['y_cn_kernel_size'], y_noise['y_cn_kernel_sigma']

    y_lambda_value = torch.tensor(1/(y_cn_kernel_sigma**2))
    y_h_value= torch.tensor(0.0, requires_grad=True)

    # State noise
    x_noise = gp['noise']

    x_wn_mu, x_wn_sigma = x_noise['x_wn_mu'], x_noise['x_wn_sigma']
    x_cn_kernel_size, x_cn_kernel_sigma, = x_noise['x_cn_kernel_size'], x_noise['x_cn_kernel_sigma']

    x_lambda_value = torch.tensor(1/(x_cn_kernel_sigma**2))
    x_h_value = torch.tensor(0.0, requires_grad=True)

    # Put all noise information in one dictionary:
    noise = {
        'y':
            {
             'y_wn_mu': y_wn_mu, 'y_wn_sigma': y_wn_sigma,
             'y_cn_kernel_size': y_cn_kernel_size, 'y_cn_kernel_sigma': y_cn_kernel_sigma,
             'y_h_value': y_h_value, 'y_lambda_value': y_lambda_value
            },
        'x':
            {
             'x_wn_mu': x_wn_mu, 'x_wn_sigma': x_wn_sigma,
             'x_cn_kernel_size': x_cn_kernel_size, 'x_cn_kernel_sigma': x_cn_kernel_sigma,
             'x_h_value': x_h_value, 'x_lambda_value': x_lambda_value
            },
    }

    (E_theta, sigma_theta,
     E_pi_x, sigma_lambda_x,
     E_pi_y, sigma_lambda_y,
     nu_x, kappa_x, lambda_eta_adapt, lambda_eta_rate, lambda_eta_t_0, lambda_eta_gamma, lambda_interval, lambda_beta,
     theta_eta_adapt, theta_eta_rate, theta_eta_t_0, theta_eta_gamma, theta_interval, theta_beta, carry_cov, jitter) = \
            (params['priors']['theta']['E_theta'],params['priors']['theta']['sigma_theta'],
             params['priors']['lambda']['E_pi_x'], params['priors']['lambda']['sigma_lambda_x'],
             params['priors']['lambda']['E_pi_y'], params['priors']['lambda']['sigma_lambda_y'],
            [torch.tensor(n) for n in opt['x']['nu']],
            opt['x']['kappa_x'],
            opt['lambda']['adapt'], opt['lambda']['eta']['rate'], opt['lambda']['eta']['t_0'], opt['lambda']['eta']['gamma'],
            opt['lambda']['inter'], opt['lambda']['beta'],
            opt['theta']['adapt'], opt['theta']['eta']['rate'], opt['theta']['eta']['t_0'], opt['theta']['eta']['gamma'],
            opt['theta']['inter'], opt['theta']['beta'],
            opt['carry_cov'], opt['jitter'])

    E_theta = [torch.tensor(v, dtype=torch.float64) for v in E_theta]
    sigma_theta = [torch.tensor(v, dtype=torch.float64) for v in sigma_theta]

    E_pi_x = [torch.tensor(v, dtype=torch.float64) for v in E_pi_x]
    sigma_lambda_x = [torch.tensor(v, dtype=torch.float64) for v in sigma_lambda_x]

    E_pi_y = [torch.tensor(v, dtype=torch.float64) for v in E_pi_y]
    sigma_lambda_y = [torch.tensor(v, dtype=torch.float64) for v in sigma_lambda_y]

    return ([kx, ky, gp_name,
            dt, T, f_name, g_name,
            E_theta, sigma_theta,
            E_pi_x, sigma_lambda_x,
            E_pi_y, sigma_lambda_y,
            nu_x, kappa_x,
            lambda_eta_adapt, lambda_eta_rate, lambda_eta_t_0, lambda_eta_gamma, lambda_interval, lambda_beta,
            theta_eta_adapt, theta_eta_rate, theta_eta_t_0, theta_eta_gamma, theta_interval, theta_beta,
            carry_cov, jitter, algorithm_name, device], noise)