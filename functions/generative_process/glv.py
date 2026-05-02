import torch
from functions.noise_generation import generate_colored_noise


# Second approach of adding noise
def glv(x, u, A, r):
    dx = x * (r + A @ x) + u
    return dx


def build(dt, T, x_noise):


    N = int(T / dt)
    a, b, c = 1.0, 2.0, 0.5

    r = torch.tensor([0., 0., 0.], dtype=torch.float64)
    A_base = torch.tensor(
        [[0.,  a, -b],
         [-a, 0.,  c],
         [b, -c, 0.]],
        dtype=torch.float64
    )

    # Amplitude scaling
    scale_s = 5.0
    A = A_base / scale_s

    # Create smooth noise
    x_white_noise, x_sigma_schedule, x_colored_noise, x_context_lengths = (
        generate_colored_noise.generate(
            (N, 3),
            x_noise['x_wn_mu'],
            x_noise['x_wn_sigma'],
            x_noise['x_cn_kernel_size'],
            x_noise['x_cn_kernel_sigma']
        )
    )
    colored_noise = torch.transpose(x_colored_noise, 0, 1)

    for add_noise in [True, False]:
        x = torch.zeros((N, 3), dtype=torch.float64)

        x0_base = torch.tensor([0.45, 0.22, 0.33], dtype=torch.float64)
        x[0] = scale_s * x0_base

        u = torch.zeros(3, dtype=torch.float64)

        for i in range(N - 1):
            xi = torch.clamp(x[i], min=0.0, max=5.0)

            if add_noise:
                u = colored_noise[i]
            else:
                u = torch.zeros_like(u)

            # ---------- INTEGRATION STEP ----------
            xn = xi + dt * glv(xi, u, A, r)
            # ------------------------------------

            xn = torch.clamp(xn, min=0.0, max=5.0)
            x[i + 1] = xn

        if add_noise:
            x_noisy = x
        else:
            x_clean = x

    return (
        x_noisy,
        x_clean,
        x_white_noise,
        x_sigma_schedule,
        x_colored_noise,
        x_context_lengths,
    )
