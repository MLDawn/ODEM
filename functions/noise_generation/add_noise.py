import torch
from functions.noise_generation import generate_white_noise, generate_colored_noise
import matplotlib.pyplot as plt
import sys
def add(x, x_wn_mu, x_wn_sigma, x_cn_kernel_size, x_cn_kernel_sigma, min_cap=None):
    '''
    :param x: The hidden state
    :type x: Torch tensor (N x dx) where N is the number of data points and dx is the dimension of each point in x
    :param mu: The mean of the white noise Gaussian
    :type mu: float
    :param sigma: The standard deviation of the white noise Gaussian
    :type sigma: float
    :param noise_type: 'color' or 'white'
    :type noise_type: string
    :param conv_kernel_size: The kernel size of the convolution kernel for creating color noise
    :type conv_kernel_size: integer
    :param conv_kernel_sigma: The standard deviation of the convolution kernel for creating color noise
    :type conv_kernel_sigma: float
    :return: Noisy observations (N x dx) where N is the number of data points and dx is the dimension of each point in x
    :rtype: Torch tensor
    '''
    # Holds the final noisy observations
    noisy_x = []
    # Holds an independent random noise tensor vector, per each dimension of x

    white_noise, sigma_schedule, colored_noise, context_lengths = generate_colored_noise.generate(x.shape, x_wn_mu, x_wn_sigma, x_cn_kernel_size, x_cn_kernel_sigma)
    noise = torch.transpose(colored_noise, 0, 1)

    for dim in range(x.shape[1]):
        # Add each noise sequence to its corresponding dimension of x. This ensures independent noise per channel
        sumed = x[:, dim] + noise[:, dim]
        if min_cap != None:
            sumed = torch.clamp(sumed, min=min_cap)
        noisy_x.append(sumed)

    noisy_x = torch.transpose(torch.stack(noisy_x), 0, 1)



    return noisy_x, white_noise, sigma_schedule, colored_noise, context_lengths