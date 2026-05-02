import torch


def step(q_theta_mu, acc_grad_theta, theta_lr):
    """
    Perform a parameter (theta) update step using accumulated gradients.

    This function applies a gradient-descent style update to the posterior
    mean of the parameters (q_theta_mu), using externally accumulated gradients.
    The update is executed under `torch.no_grad()` to prevent PyTorch from
    tracking this operation in the computation graph.

    After the update, the tensor is detached and re-enabled for gradient
    tracking to allow subsequent gradient accumulation in later steps.

    Parameters
    ----------
    q_theta_mu
        Posterior mean of the model parameters (theta).
    acc_grad_theta
        Accumulated gradient of the variational free energy with respect to
        q_theta_mu (typically collected over multiple D-steps).
    theta_lr
        Learning rate for the parameter update.

    Returns
    -------
    q_theta_mu
        Updated parameter posterior mean with gradient tracking re-enabled.

    Notes
    -----
    - This function assumes gradients have already been accumulated elsewhere
      (e.g., during the D-step loop).
    - Detaching and re-attaching `requires_grad` ensures a clean computation
      graph for the next accumulation phase.
    """

    # Apply the parameter update without tracking gradients
    # (this is a manual optimisation step, not part of autograd).
    with torch.no_grad():
        q_theta_mu = q_theta_mu - theta_lr * acc_grad_theta

    # Detach from the old graph and re-enable gradient tracking
    # for the next round of gradient accumulation.
    q_theta_mu = q_theta_mu.detach().requires_grad_(True)

    return q_theta_mu
