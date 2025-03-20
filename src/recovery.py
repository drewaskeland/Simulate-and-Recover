#Code produced with Chat GPT assistance

import numpy as np

def recover_parameters(R_obs, M_obs, V_obs):
    """
    Recover EZ diffusion model parameters (a, v, t) from observed summary statistics.
    
    Assumes the forward functions are defined as:
        y = exp(-a*v)
        R = 1/(1+y)
        M = t + (a/(2*v)) * ((1-y)/(1+y))
        V = (a/(2*v**3)) * ((1 - 2*a*v*y - y**2)/((1+y)**2))
    
    The inversion is carried out by:
      1. Computing y from the accuracy equation.
      2. Setting A = a*v = -ln(y).
      3. Solving for r = a/v via:
             r^2 = (2*A * V_obs * (1+y)**2) / (1 - 2*A*y - y**2).
         If r^2 is negative or non-finite (due to simulation noise), we set it to a small positive value.
      4. Then a = sqrt(A * r) and v = sqrt(A / r).
      5. Finally, t = M_obs - (r/2)*((1-y)/(1+y)).
    
    Parameters:
        R_obs (float): Observed accuracy (between 0 and 1, exclusive).
        M_obs (float): Observed mean response time.
        V_obs (float): Observed variance of response times.
    
    Returns:
        dict: A dictionary with keys "a", "v", and "t".
    """
    # Validate and convert inputs.
    try:
        R_obs = float(R_obs)
        M_obs = float(M_obs)
        V_obs = float(V_obs)
    except Exception:
        raise ValueError("Inputs must be numeric.")
        
    if R_obs <= 0.0 or R_obs >= 1.0:
        raise ValueError("R_obs must be strictly between 0 and 1.")
    
    # Clip R_obs slightly to avoid numerical issues.
    epsilon = 1e-5
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)
    
    # If R_obs is too close to 0.5, nudge it slightly.
    threshold = 1e-3
    if np.abs(R_obs - 0.5) < threshold:
        R_obs = 0.5 + threshold if R_obs >= 0.5 else 0.5 - threshold

    # Ensure V_obs is positive.
    if V_obs <= 1e-8:
        V_obs = 1e-8

    # Step 1: Recover y from R_obs.
    # From R = 1/(1+y)  =>  y = 1/R - 1.
    y = 1.0 / R_obs - 1.0

    # Step 2: Compute A = a*v.
    A = -np.log(y)
    if not np.isfinite(A) or A <= 0:
        A = 1e-8

    # Step 3: Invert the variance equation to get r^2.
    # r^2 = (2*A * V_obs * (1+y)**2) / (1 - 2*A*y - y**2)
    denom = 1 - 2 * A * y - y**2
    denom = np.maximum(denom, 1e-8)  # avoid division by zero
    r2 = (2 * A * V_obs * (1 + y)**2) / denom
    # If r2 is negative or not finite, force it to a small positive value.
    if (r2 < 0) or (not np.isfinite(r2)):
        r2 = 1e-8
    r = np.sqrt(r2)
    if (not np.isfinite(r)) or (r <= 0):
        r = 1e-4

    # Step 4: Recover a and v.
    a_est = np.sqrt(A * r)
    v_est = np.sqrt(A / r)
    if (not np.isfinite(a_est)) or (a_est <= 0):
        a_est = 1e-4
    if (not np.isfinite(v_est)) or (v_est <= 0):
        v_est = 1e-4

    # Step 5: Recover t.
    t_est = M_obs - (r / 2) * ((1 - y) / (1 + y))
    if not np.isfinite(t_est):
        t_est = 0.0
    
    return a_est, v_est, t_est


if __name__ == "__main__":
    # For testing, assume noise-free observed summary statistics generated
    # with true parameters a=1, v=1, t=0.3.
    # Using the forward function:
    #   y = exp(-1) ~ 0.3679,
    #   R ~ 1/(1+0.3679) ~ 0.7311,
    #   M ~ 0.3 + (1/(2*1)) * ((1-0.3679)/(1+0.3679)) ~ 0.5311,
    #   V computed accordingly (e.g., ~0.0345).
    R_obs = 0.7311
    M_obs = 0.5311
    V_obs = 0.0345
    
    try:
        recovered = recover_parameters(R_obs, M_obs, V_obs)
        print(f"Recovered parameters: a={recovered['a']:.3f}, v={recovered['v']:.3f}, t={recovered['t']:.3f}")
    except ValueError as e:
        print("Error:", e)
