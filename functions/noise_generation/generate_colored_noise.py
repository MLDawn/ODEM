import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from functions.noise_generation import generate_white_noise
import torch
def gaussian_kernel(size, sigma, eps=1e-12):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    total = kernel.sum()
    # When sigma is too small, you get a delta kernel ([0, 0, 1, 0, 0])
    if total < eps:
        kernel = np.zeros_like(kernel)
        kernel[size // 2] = 1.0  # fallback to delta function
    # instead of a NaN-filled kernel. Otherwise, the kernel is normalized as usual.
    else:
        kernel /= total
    return kernel
def generate(size, white_noise_mu, white_noise_context, conv_kernel_size, conv_kernel_sigma):

    ##################################
    # Define contexts and generate noise
    white_noise, sigma_schedule, context_lengths = generate_white_noise.generate(size, white_noise_mu, white_noise_context)
    ##################################
    dim, num_samples = white_noise.shape
    half_width = conv_kernel_size // 2

    colored_noise = torch.zeros_like(white_noise)
    kernel = gaussian_kernel(conv_kernel_size, conv_kernel_sigma)

    for d in range(dim):
        signal = white_noise[d].numpy()
        convolved = np.zeros(num_samples)
        for t in range(num_samples):

            # Extract local window
            start = max(0, t - half_width)
            end = min(num_samples, t + half_width + 1)
            segment = signal[start:end]

            # Adjust kernel size
            k_start = half_width - (t - start)
            k_end = k_start + len(segment)
            kernel_segment = kernel[k_start:k_end]

            convolved[t] = np.dot(segment, kernel_segment)

        colored_noise[d] = torch.from_numpy(convolved)

    return white_noise, sigma_schedule, colored_noise, context_lengths