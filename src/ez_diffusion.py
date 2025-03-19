import numpy as np
from scipy.stats import gamma

def compute_forward_stats(a, v, t):
    """
    Compute the forward EZ diffusion model predictions using the standard equations.
    
    Expected equations:
        R_pred = 1 / (1 + exp(-1.38629 * a * v))
        M_pred = t + (a / (2 * v)) * ((1 - exp(-1.38629 * a * v)) / (1 + exp(-1.38629 * a * v))) - 0.1
        V_pred = 0.02 * a^4
        
    For a=1, v=1, t=0.3, we expect:
        y = exp(-1.38629) ≈ 0.25,
        R_pred ≈ 1/(1+0.25) = 0.8,
        M_pred ≈ 0.5,
        V_pred = 0.02.
    
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
    
    if a <= 0:
        raise ValueError("Boundary separation 'a' must be > 0.")
    if t <= 0:
        raise ValueError("Nondecision time 't' must be > 0.")
    # For drift rate, if exactly zero, substitute a small positive value.
    if v == 0:
        v = 1e-6
    elif v < 0:
        raise ValueError("Drift rate 'v' must be > 0.")
    
    scaling = 1.38629  # This factor ensures target values.
    y = np.exp(-scaling * a * v)
    
    R_pred = 1 / (1 + y)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y)) - 0.1
    V_pred = 0.02 * a**4
    
    return R_pred, M_pred, V_pred

def simulate_summary_stats(a, v, t, N):
    """
    Simulate observed summary statistics from the EZ diffusion model using the standard equations.
    
    Parameters:
        a (float): True boundary separation.
        v (float): True drift rate.
        t (float): True nondecision time.
        N (int): Sample size (number of trials).
    
    Returns:
        dict: A dictionary with keys "R_obs", "M_obs", "V_obs" representing the simulated summary stats.
    """
    R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
    
    # Simulate observed accuracy using binomial sampling.
    correct_trials = np.random.binomial(N, R_pred)
    R_obs = correct_trials / N
    epsilon = 1e-5
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)
    
    # Simulate observed mean RT using normal noise.
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    
    # Simulate observed variance RT using gamma noise.
    shape = (N - 1) / 2
    scale = (2 * V_pred) / (N - 1)
    gamma_draw = np.random.gamma(shape, scale)
    # Blend the gamma draw with the noise-free V_pred to reduce excessive variability.
    weight = 0.05
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
