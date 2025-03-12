import numpy as np

def recover_parameters(R_obs, M_obs, V_obs):
    """
    Recover EZ diffusion model parameters from observed summary statistics.

    Parameters:
        R_obs (float): Observed accuracy rate.
        M_obs (float): Observed mean response time.
        V_obs (float): Observed variance of response times.

    Returns:
        nu_est (float): Estimated drift rate.
        a_est (float): Estimated boundary separation.
        t_est (float): Estimated nondecision time.
    """
    # Clip R_obs to avoid 0 or 1
    epsilon = 1e-5
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)

    # If R_obs is too close to chance (0.5), adjust so L does not throw errors
    threshold = 1e-3
    if np.abs(R_obs - 0.5) < threshold:
        # If R_obs is exactly 0.5 (or nearly), push it away from chance
        R_obs = 0.5 + threshold if R_obs >= 0.5 else 0.5 - threshold

    # Compute L = ln(R_obs / (1 - R_obs))
    L = np.log(R_obs / (1 - R_obs))

    # Inverse equation for drift rate (nu)
    sign_factor = np.sign(R_obs - 0.5)
    inside = L * (R_obs ** 2 * L - R_obs * L + R_obs - 0.5)
    nu_est = sign_factor * (np.abs(inside) / V_obs) ** 0.25

    # Now compute boundary separation (a)
    a_est = L / nu_est

    # And compute nondecision time (tau)
    t_est = M_obs - (a_est / (2 * nu_est)) * ((1 - np.exp(-a_est * nu_est)) / (1 + np.exp(-a_est * nu_est)))

    return nu_est, a_est, t_est

if __name__ == "__main__":
    # Example usage with some observed summary statistics:
    R_obs = 0.73  # Example observed accuracy rate
    M_obs = 0.56  # Example observed mean response time
    V_obs = 0.034  # Example observed variance of response times

    nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)

    print(f"Recovered parameters: nu_est={nu_est:.3f}, a_est={a_est:.3f}, t_est={t_est:.3f}")
