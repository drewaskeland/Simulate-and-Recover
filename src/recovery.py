import numpy as np

def recover_parameters(accuracy_obs, mean_RT_obs, variance_RT_obs):
    """
    Recover EZ diffusion model parameters from observed summary statistics.

    Parameters:
        accuracy_obs (float): Observed accuracy (should be strictly between 0 and 1).
        mean_RT_obs (float): Observed mean response time.
        variance_RT_obs (float): Observed variance of response times.

    Returns:
        tuple: (drift_est, boundary_est, non_decision_est)
            - drift_est (float): Recovered drift rate.
            - boundary_est (float): Recovered boundary separation.
            - non_decision_est (float): Recovered non-decision time.
    """
    # We assume accuracy_obs is generated via a truncated normal, so it is already within (0, 1).
    if accuracy_obs <= 0.0 or accuracy_obs >= 1.0:
        raise ValueError("accuracy_obs must be between 0 and 1 (exclusive).")
    
    # Adjust values extremely close to chance to avoid numerical instability.
    thresh = 1e-3
    if np.abs(accuracy_obs - 0.5) < thresh:
        accuracy_obs = 0.5 + thresh if accuracy_obs >= 0.5 else 0.5 - thresh

    log_ratio = np.log(accuracy_obs / (1 - accuracy_obs))
    sign_val = np.sign(accuracy_obs - 0.5)
    
    expression = log_ratio * (accuracy_obs**2 * log_ratio - accuracy_obs * log_ratio + accuracy_obs - 0.5)
    
    drift_est = sign_val * (expression / variance_RT_obs)**0.25
    boundary_est = log_ratio / drift_est
    non_decision_est = mean_RT_obs - (boundary_est / (2 * drift_est)) * (
        (1 - np.exp(-boundary_est * drift_est)) / (1 + np.exp(-boundary_est * drift_est))
    )
    
    return drift_est, boundary_est, non_decision_est

# Example Usage
if __name__ == "__main__":
    # Example observed summary statistics from your simulation (which uses a truncated normal for accuracy)
    sample_data = (0.75, 0.5, 0.04)
    drift_est, boundary_est, non_decision_est = recover_parameters(*sample_data)
    print("Recovered drift rate:", drift_est)
    print("Recovered boundary separation:", boundary_est)
    print("Recovered non-decision time:", non_decision_est)
