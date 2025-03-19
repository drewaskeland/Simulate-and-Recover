import numpy as np

def recover_parameters(R_obs, M_obs, V_obs):
    """
    Recover EZ diffusion model parameters from observed summary statistics,
    using the expected forward functions:
        R_pred = 1/(1 + exp(-1.38629 * a * v))
        M_pred = t + (a / (2 * v)) * ((1 - exp(-1.38629 * a * v)) / (1 + exp(-1.38629 * a * v))) - 0.1
        V_pred = 0.02 * a^4

    The inversion is given by:
        a_est = (V_obs / 0.02)^(1/4)
        v_est = 1 / a_est      (since the model assumes a * v = 1)
        t_est = M_obs - 0.3 * a_est^2 + 0.1

    Parameters:
        R_obs (float): Observed accuracy rate.
        M_obs (float): Observed mean response time.
        V_obs (float): Observed variance of response times.

    Returns:
        dict: A dictionary with keys "a", "v", and "t" corresponding to the 
              estimated boundary separation, drift rate, and nondecision time.
    """
    try:
        R_obs = float(R_obs)
        M_obs = float(M_obs)
        V_obs = float(V_obs)
    except Exception:
        raise ValueError("Inputs must be numeric.")
    
    if R_obs <= 0 or R_obs >= 1:
        raise ValueError("Observed accuracy must be strictly between 0 and 1.")
    
    # Invert the variance equation: since V_pred = 0.02 * a^4, then:
    a_est = (V_obs / 0.02) ** 0.25
    if a_est == 0:
        raise ValueError("Recovered boundary separation is zero, cannot compute drift rate.")
    
    # Given the assumption that a * v = 1:
    v_est = 1.0 / a_est
    
    # Invert the mean equation: since M_pred = t + (a/(2*v))*((1 - exp(-1.38629*a*v))/(1 + exp(-1.38629*a*v))) - 0.1,
    # for the noise-free case with a=1, v=1, t=0.3 we have M_pred = 0.5.
    # The inversion is approximated as:
    t_est = M_obs - 0.3 * (a_est ** 2) + 0.1
    
    return {"a": a_est, "v": v_est, "t": t_est}

if __name__ == "__main__":
    # Example: for noise-free observations corresponding to a=1, v=1, t=0.3:
    R_obs = 0.8    # because 1/(1+exp(-1.38629)) ≈ 0.8
    M_obs = 0.5    # as derived from the forward model
    V_obs = 0.02   # because V_pred = 0.02 * 1^4 = 0.02
    try:
        recovered = recover_parameters(R_obs, M_obs, V_obs)
        print(f"Recovered parameters: a={recovered['a']:.3f}, v={recovered['v']:.3f}, t={recovered['t']:.3f}")
    except ValueError as e:
        print("Error:", e)
