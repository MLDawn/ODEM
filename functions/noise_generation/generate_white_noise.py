import numpy as np
import torch
def exp_schedule(start, end, steps):
    ratio = np.linspace(0, 1, steps)
    return start * (end / start) ** ratio

def sigmoid_schedule(start, end, steps, steepness=0.3):
    x = np.linspace(-30, 30, steps)
    s = 1 / (1 + np.exp(-steepness * x))
    return start + (end - start) * s

def linear_schedule(start, end, steps):
    return np.linspace(start, end, steps)

def log_schedule(start, end, steps):
    return np.logspace(np.log10(start), np.log10(end), steps)

def gaussian_schedule(start, end, steps, invert=False):
    x = np.linspace(-1, 1, steps)
    bump = np.exp(-x**2 / (2 * 0.5**2))  # width controlled by 0.2
    bump = (bump - bump.min()) / (bump.max() - bump.min())  # normalize to [0,1]
    if invert:
        bump = 1 - bump  # upside-down Gaussian
    return start + (end - start) * bump

def generate(size, mu, contexts):
    np.random.seed(seed=42)  # This is needed, even though we are Seeding in main.py

    num_samples, dim = size
    num_contexts = len(contexts)
    context_lengths = [num_samples // num_contexts] * num_contexts
    context_lengths[-1] += num_samples - sum(context_lengths)

    sigma_schedule = []
    for (sigma_start, sigma_end, mode), length in zip(contexts, context_lengths):
        if mode == 'exp':
            sigmas = exp_schedule(sigma_start, sigma_end, length)
        elif mode == 'sigmoid':
            sigmas = sigmoid_schedule(sigma_start, sigma_end, length)
        elif mode == 'linear':
            sigmas = linear_schedule(sigma_start, sigma_end, length)
        elif mode == 'log':
            sigmas = log_schedule(sigma_start, sigma_end, length)
        elif mode == 'gaussian':
            sigmas = gaussian_schedule(sigma_start, sigma_end, length)
        else:
            raise ValueError(f"Unknown scheduling mode: {mode}")
        sigma_schedule.extend(sigmas)

    sigma_schedule = np.array(sigma_schedule)

    white_noise = []
    for _ in range(dim):
        noise = np.random.normal(loc=mu, scale=sigma_schedule)
        white_noise.append(torch.from_numpy(noise))

    white_noise = torch.stack(white_noise)  # Shape: (dim, num_samples)
    context_lengths = np.array(context_lengths)


    return white_noise, sigma_schedule, context_lengths

# def generate(size, mu, sigma):
#     num_samples, dim = size[0], size[1]
#     white_noise = []
#     for _ in range(dim):
#         # num_samples number of noisy samples generated PER dimension, independently.
#         white_noise.append(torch.from_numpy(np.random.normal(size=num_samples, loc=mu, scale=sigma)))
#
#     white_noise = torch.stack(white_noise)
#     return white_noise