import torch


def step(q_lambda_x_mu, q_lambda_y_mu, acc_grad_lambda_x, acc_grad_lambda_y, lambda_lr):
    """
    Perform a precision (lambda) update step using accumulated gradients.

    This function updates the posterior means of the log-precision
    hyperparameters for:
      - state noise  (q_lambda_x_mu)
      - observation noise (q_lambda_y_mu)

    The updates are performed using accumulated gradients and a shared
    learning rate, under `torch.no_grad()` to avoid contaminating the
    autograd computation graph.

    After the update, both tensors are detached and re-enabled for
    gradient tracking so that new gradients can be accumulated in
    subsequent inference steps.

    Parameters
    ----------
    q_lambda_x_mu
        Posterior mean of the log-precision hyperparameters for state noise.
    q_lambda_y_mu
        Posterior mean of the log-precision hyperparameters for observation noise.
    acc_grad_lambda_x
        Accumulated gradient of the variational free energy with respect to
        q_lambda_x_mu.
    acc_grad_lambda_y
        Accumulated gradient of the variational free energy with respect to
        q_lambda_y_mu.
    lambda_lr
        Learning rate for the precision (lambda) updates.

    Returns
    -------
    q_lambda_x_mu
        Updated posterior mean of the state-noise log-precision hyperparameters,
        with gradient tracking re-enabled.
    q_lambda_y_mu
        Updated posterior mean of the observation-noise log-precision hyperparameters,
        with gradient tracking re-enabled.

    Notes
    -----
    - This function assumes gradients are accumulated elsewhere (e.g., during
      repeated D-steps) before being applied here.
    - Detaching and re-attaching `requires_grad` ensures clean gradient graphs
      for the next accumulation phase.
    """

    # Apply precision updates without tracking gradients
    # (manual optimisation step outside autograd).
    with torch.no_grad():
        q_lambda_x_mu = q_lambda_x_mu - lambda_lr * acc_grad_lambda_x
        q_lambda_y_mu = q_lambda_y_mu - lambda_lr * acc_grad_lambda_y

    # Detach from the previous computation graph and re-enable gradient tracking
    # for subsequent gradient accumulation.
    q_lambda_x_mu = q_lambda_x_mu.detach().requires_grad_(True)
    q_lambda_y_mu = q_lambda_y_mu.detach().requires_grad_(True)

    return q_lambda_x_mu, q_lambda_y_mu



