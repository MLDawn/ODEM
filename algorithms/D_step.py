"""
This module implements a single D-step update routine.

The D-step (in your DEM / generalised predictive-coding style loop) performs fast
inference over the generalised hidden-state posterior mean q_x_mu by:

1) Computing the current variational free energy (VFE) given the current beliefs.
2) Checking VFE validity for numerical pathologies (e.g., NaN/Inf).
3) Updating q_x_mu using Ozaki integration.

Important
---------
- This file intentionally contains no parameter learning (theta) and no precision learning (lambdas);
  it only updates the hidden-state belief q_x_mu.
- The function below assumes q_x_mu has shape (k_x, d_x):
    k_x = number of generalised coordinates of motion
    d_x = hidden-state dimensionality
"""

from functions.vfe_calculation import compute_vfe, vfe_validity
from functions.optimisation import ozaki


def step(q_x_mu, gen_y, f, g, q_theta_mu, p_theta_eta, q_lambda_x_mu, p_lambda_x_eta,
                              q_lambda_y_mu, p_lambda_y_eta,
                              gen_pi_y, gen_pi_x, p_theta_pi, gen_mu_cov, q_theta_cov,
                              p_lambda_x_pi, q_lambda_x_cov,
                              p_lambda_y_pi, q_lambda_y_cov, kappa_x, nu_x, initial_jitter, D, idx, device):
    """
    Perform one D-step state-inference update.

    This function:
      - computes the current VFE for the provided beliefs,
      - validates that the VFE is numerically well-defined,
      - applies an Ozaki integration step to update q_x_mu (generalised hidden-state mean).

    Parameters
    ----------
    q_x_mu
        Posterior mean of the generalised hidden state, shaped (k_x, d_x).
    gen_y
        Generalised observations at the current time step.
    f
        Generative dynamics function used by the model (state transition in continuous time / generalised coordinates).
    g
        Generative observation function used by the model (mapping hidden states to observations).
    q_theta_mu
        Posterior mean of model parameters (theta).
    p_theta_eta
        Prior mean of model parameters (theta).
    q_lambda_x_mu
        Posterior mean of state-noise log-precision hyperparameters.
    p_lambda_x_eta
        Prior mean of state-noise log-precision hyperparameters.
    q_lambda_y_mu
        Posterior mean of observation-noise log-precision hyperparameters.
    p_lambda_y_eta
        Prior mean of observation-noise log-precision hyperparameters.
    gen_pi_y
        Generalised observation-noise precision structure used inside VFE.
    gen_pi_x
        Generalised state-noise precision structure used inside VFE.
    p_theta_pi
        Prior precision over theta used inside VFE.
    gen_mu_cov
        Generalised state covariance structure (as used in your VFE computation).
    q_theta_cov
        Posterior covariance of theta (as used in your VFE computation).
    p_lambda_x_pi
        Prior precision over state log-precision hyperparameters.
    q_lambda_x_cov
        Posterior covariance of state log-precision hyperparameters.
    p_lambda_y_pi
        Prior precision over observation log-precision hyperparameters.
    q_lambda_y_cov
        Posterior covariance of observation log-precision hyperparameters.
    kappa_x
        State inference step size / gain used by the Ozaki integrator.
    nu_x
        Additional integration / damping hyperparameter used by the Ozaki integrator.
    initial_jitter
        Initial stabilising jitter value used inside Ozaki integration (e.g., for matrix inversions / logdets).
    D
        Generalised derivative operator (or related operator) required by the Ozaki integrator.
    idx
        Current time-step index (used for informative VFE validity messages).
    device
        Target compute device (e.g., torch.device('cuda') / 'cpu') used inside VFE and Ozaki.

    Returns
    -------
    q_x_mu
        Updated posterior mean of the generalised hidden state, shaped (k_x, d_x).
    delta_s
        Diagnostic / integration step quantity returned by ozaki.integrate (exact meaning defined in your implementation).
    ozaki_jitter
        Jitter value actually used/selected by ozaki.integrate (may be adapted from initial_jitter).

    Notes
    -----
    - This function does not modify theta or lambda beliefs; it only updates q_x_mu.
    - VFE is computed before integration and validated to catch instabilities early.
    """

    # q_x_mu is assumed to have shape (k_x, d_x):
    #   kx = number of generalised coordinates of motion
    #   dx = dimensionality of the hidden state vector
    kx, dx = q_x_mu.shape[0], q_x_mu.shape[1]

    # Compute variational free energy (VFE) under the current beliefs.
    # The compute() function returns (vfe, <other outputs>, <other outputs>),
    # but this step only uses vfe for validation and as input to Ozaki integration.
    vfe, _, _ = compute_vfe.compute(q_x_mu, gen_y, f, g, q_theta_mu, p_theta_eta, q_lambda_x_mu, p_lambda_x_eta,
                              q_lambda_y_mu, p_lambda_y_eta,
                              gen_pi_y, gen_pi_x, p_theta_pi, gen_mu_cov, q_theta_cov,
                              p_lambda_x_pi, q_lambda_x_cov,
                              p_lambda_y_pi, q_lambda_y_cov, device)

    # Ensure VFE is finite / numerically valid before proceeding (fail fast on NaN/Inf).
    vfe_validity.check(vfe, step_info=f"D-step t={idx}")

    # Update the generalised hidden-state posterior mean using Ozaki integration.
    # Returns:
    #   q_x_mu       updated state mean (k_x, d_x)
    #   delta_s      integration diagnostic (meaning defined in your ozaki implementation)
    #   ozaki_jitter jitter used by the integrator (may be adapted from initial_jitter)
    q_x_mu, delta_s, ozaki_jitter = ozaki.integrate(vfe, q_x_mu, kappa_x, nu_x, initial_jitter, device, D, kx, dx)

    return q_x_mu, delta_s, ozaki_jitter
