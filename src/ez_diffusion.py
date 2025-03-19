import numpy as np
from scipy.stats import norm, gamma, truncnorm

def compute_forward_stats(boundary, drift, non_decision):
    """
    Compute the predicted summary statistics of the EZ diffusion model.

    Parameters:
        boundary (float): True boundary separation.
        drift (float): True drift rate.
        non_decision (float): True non-decision time.

    Returns:
        tuple: (accuracy_pred, mean_RT_pred, variance_RT_pred)
            accuracy_pred (float): Predicted accuracy rate.
            mean_RT_pred (float): Predicted mean response time.
            variance_RT_pred (float): Predicted variance of response times.
    """
    if boundary <= 0:
        raise ValueError("Boundary separation 'boundary' must be > 0.")
    if drift <= 0:
        raise ValueError("Drift rate 'drift' must be > 0.")
    if non_decision <= 0:
        raise ValueError("Non-decision time 'non_decision' must be > 0.")

    # Compute intermediary variable.
    y = np.exp(-boundary * drift)
    
    # Equation (1): Predicted accuracy.
    accuracy_pred = 1 / (1 + y)
    
    # Equation (2): Predicted mean response time.
    mean_RT_pred = non_decision + (boundary / (2 * drift)) * ((1 - y) / (1 + y))
    
    # Equation (3): Predicted variance of response times.
    variance_RT_pred = (boundary / (2 * (drift**3))) * ((1 - 2 * boundary * drift * y - y**2) / ((1 + y)**2))
    
    return accuracy_pred, mean_RT_pred, variance_RT_pred

def simulate_summary_stats(boundary, drift, non_decision, N):
    """
    Simulate observed summary statistics from the EZ diffusion model.
    
    Parameters:
        N (int): Sample size (number of trials).
    
    Returns:
        tuple: (accuracy_obs, mean_RT_obs, variance_RT_obs)
            accuracy_obs (float): Observed accuracy rate.
            mean_RT_obs (float): Observed mean response time.
            variance_RT_obs (float): Observed variance of response times.
    """
    accuracy_pred, mean_RT_pred, variance_RT_pred = compute_forward_stats(boundary, drift, non_decision)
    
    # Simulate observed accuracy from a truncated normal distribution.
    # Use the binomial variance approximation for p:
    # Variance for proportion: p*(1-p)/N, then standard deviation is sqrt(p*(1-p)/N)
    sd_accuracy = np.sqrt((accuracy_pred * (1 - accuracy_pred)) / N)
    # Define truncation boundaries (0,1) in standardized units:
    a, b = (0 - accuracy_pred) / sd_accuracy, (1 - accuracy_pred) / sd_accuracy
    accuracy_obs = truncnorm.rvs(a, b, loc=accuracy_pred, scale=sd_accuracy)

    # Simulate the observed mean response time.
    mean_RT_obs = np.random.normal(mean_RT_pred, np.sqrt(variance_RT_pred / N))
    
    # Simulate the observed variance using a gamma distribution.
    shape = (N - 1) / 2
    scale = (2 * variance_RT_pred) / (N - 1)
    variance_RT_obs = np.random.gamma(shape, scale)
    
    return accuracy_obs, mean_RT_obs, variance_RT_obs

# Example Usage
if __name__ == "__main__":
    # Example parameters:
    boundary_true = 1.0      # Boundary separation between 0.5 and 2.
    drift_true = 1.0         # Drift rate between 0.5 and 2.
    non_decision_true = 0.3  # Non-decision time between 0.1 and 0.5.
    N = 100                  # Sample size.
    
    accuracy_pred, mean_RT_pred, variance_RT_pred = compute_forward_stats(boundary_true, drift_true, non_decision_true)
    print(f"Predicted: Accuracy={accuracy_pred:.3f}, Mean RT={mean_RT_pred:.3f}, Variance RT={variance_RT_pred:.3f}")
    
    accuracy_obs, mean_RT_obs, variance_RT_obs = simulate_summary_stats(boundary_true, drift_true, non_decision_true, N)
    print(f"Observed: Accuracy={accuracy_obs:.3f}, Mean RT={mean_RT_obs:.3f}, Variance RT={variance_RT_obs:.3f}")
