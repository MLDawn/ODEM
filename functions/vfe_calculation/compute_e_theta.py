def compute(q_theta_mu, p_theta_eta):
    e_theta = q_theta_mu - p_theta_eta#[a - b for a, b in zip(q_theta_mu, p_theta_eta)]
    return e_theta