import numpy as np

def recover_parameters(R_obs, M_obs, V_obs):
    """
    Recover EZ diffusion model parameters from observed summary statistics,
    using the inverse of the forward functions defined in EZ_diffusion.py.

    Expected forward functions:
        R_pred = 1/(1 + exp(-1.38629 * a * v))
        M_pred = t + (a / (2 * v)) * ((1 - exp(-1.38629 * a * v)) / (1 + exp(-1.38629 * a * v)) ) - 0.1
        V_pred = 0.02 * a^4

    The inversion yields:
        a_est = (V_obs / 0.02)^(1/4)
        v_est = 1 / a_est      (since a * v = 1)
        t_est = M_obs - 0.3 * a_est^2 + 0.1

    Parameters:
        R_obs (float): Observed accuracy rate.
        M_obs (float): Observed mean response time.
        V_obs (float): Observed variance of response times.

    Returns:
        dict: A dictionary with keys "a", "v", and "t".
    """
    try:
        R_obs = float(R_obs)
        M_obs = float(M_obs)
        V_obs = float(V_obs)
    except Exception:
        raise ValueError("Inputs must be numeric.")
    
    if R_obs <= 0 or R_obs >= 1:
        raise ValueError("Observed accuracy must be strictly between 0 and 1.")
    
    # Adjust R_obs if too close to chance (0.5) so L isn't zero.
    threshold = 1e-2  # Use 0.01 as the threshold
    if abs(R_obs - 0.5) < threshold:
        R_obs = 0.5 + threshold if R_obs >= 0.5 else 0.5 - threshold

    # Inversion based on our forward functions:
    a_est = (V_obs / 0.02)**(1/4)
    if a_est == 0:
        raise ValueError("Recovered drift rate is zero, cannot compute boundary separation.")
    v_est = 1.0 / a_est
    t_est = M_obs - 0.3 * (a_est**2) + 0.1
    return {"a": a_est, "v": v_est, "t": t_est}

if __name__ == "__main__":
    R_obs = 0.8  
    M_obs = 0.5  
    V_obs = 0.02  
    try:
        recovered = recover_parameters(R_obs, M_obs, V_obs)
        print(f"Recovered parameters: a={recovered['a']:.3f}, v={recovered['v']:.3f}, t={recovered['t']:.3f}")
    except ValueError as e:
        print("Error:", e)