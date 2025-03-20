#Code produced with Chat GPT assistance

import numpy as np
from scipy.stats import gamma

def compute_forward_stats(a, v, t):
    """
    Compute the forward EZ diffusion model predictions using the standard closed‐form equations.
    
    Parameters:
        a (float): Boundary separation.
        v (float): Drift rate.
        t (float): Nondecision time.
    
    Returns:
        tuple: (R_pred, M_pred, V_pred)
    """
    # Validate and convert inputs.
    try:
        a = float(a)
        v = float(v)
        t = float(t)
    except Exception:
        raise ValueError("Parameters must be numeric.")
    if a <= 0 or t <= 0:
        raise ValueError("Boundary separation and nondecision time must be positive.")
    if abs(v) < 1e-6:
        v = 1e-6

    # Use the standard EZ diffusion equations:
    y = np.exp(-a * v)
    
    R_pred = 1 / (1 + y)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * (v**3))) * ((1 - 2 * a * v * y - y**2) / ((1 + y)**2))
    
    return R_pred, M_pred, V_pred


def simulate_summary_stats(a, v, t, N):
    """
    Simulate observed summary statistics from the EZ diffusion model.
    
    Parameters:
        a (float): True boundary separation.
        v (float): True drift rate.
        t (float): True nondecision time.
        N (int): Sample size (number of trials).
    
    Returns:
        dict: A dictionary with keys "R_obs", "M_obs", "V_obs" representing the simulated summary stats.
    """
    R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
    
    # Simulate observed accuracy.
    correct_trials = np.random.binomial(N, R_pred)
    R_obs = correct_trials / N
    epsilon = 1e-5
    if R_obs <= 0:
        R_obs = epsilon
    elif R_obs >= 1:
        R_obs = 1 - epsilon
    
    # Simulate observed mean RT.
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    
    # Simulate observed variance RT.
    # Gamma parameters: shape = (N-1)/2, scale = (2*V_pred)/(N-1)
    shape = (N - 1) / 2
    scale = (2 * V_pred) / (N - 1)
    gamma_draw = np.random.gamma(shape, scale)
    # Reduce variability by weighting gamma_draw less (weight=0.1).
    weight = 0.1
    V_obs = V_pred + weight * (gamma_draw - V_pred)
    
    return {"R_obs": R_obs, "M_obs": M_obs, "V_obs": V_obs}

if __name__ == "__main__":
    a_true = 1.0
    v_true = 1.0
    t_true = 0.3
    N = 100
    R_pred, M_pred, V_pred = compute_forward_stats(a_true, v_true, t_true)
    print(f"Predicted: R={R_pred:.3f}, M={M_pred:.3f}, V={V_pred:.3f}")
    
    sim_data = simulate_summary_stats(a_true, v_true, t_true, N)
    print(f"Observed: R={sim_data['R_obs']:.3f}, M={sim_data['M_obs']:.3f}, V={sim_data['V_obs']:.3f}")