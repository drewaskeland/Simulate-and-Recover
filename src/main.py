#Code produced with ChatGPT assistance

import numpy as np
from ez_diffusion import simulate_summary_stats
from recovery import recover_parameters

def simulate_and_estimate(sample_size, iterations=1000):
    """
    Run a simulation and parameter estimation loop over a given number of iterations.
    In each iteration, random true parameters are drawn, summary statistics are simulated,
    and parameters are recovered.

    Parameters:
        sample_size (int): Number of trials.
        iterations (int): Number of simulation iterations.

    Returns:
        tuple: (avg_bias, avg_squared_error, valid_iterations, invalid_iterations)
            - avg_bias: Mean bias for [drift, boundary, non_decision] across valid iterations.
            - avg_squared_error: Mean squared error for [drift, boundary, non_decision] across valid iterations.
            - valid_iterations: Count of iterations with successful recovery.
            - invalid_iterations: Count of iterations where recovery failed.
    """
    biases = []
    squared_errors = []
    failed_count = 0

    for _ in range(iterations):
        # Randomly sample true parameters for this iteration.
        boundary_true = np.random.uniform(0.5, 2)
        drift_true = np.random.uniform(0.5, 2)
        non_decision_true = np.random.uniform(0.1, 0.5)
        
        # Simulate observed summary statistics.
        R_obs, M_obs, V_obs = simulate_summary_stats(boundary_true, drift_true, non_decision_true, sample_size)
        
        try:
            drift_est, boundary_est, non_decision_est = recover_parameters(R_obs, M_obs, V_obs)
        except ValueError:
            # Append NaNs when recovery fails (e.g., due to extreme observed values 1 or 0).
            biases.append(np.array([np.nan, np.nan, np.nan]))
            squared_errors.append(np.array([np.nan, np.nan, np.nan]))
            failed_count += 1
            continue
        
        # Check if any recovered parameter is NaN.
        if np.isnan(drift_est) or np.isnan(boundary_est) or np.isnan(non_decision_est):
            biases.append(np.array([np.nan, np.nan, np.nan]))
            squared_errors.append(np.array([np.nan, np.nan, np.nan]))
            failed_count += 1
        else:
            current_bias = np.array([drift_true, boundary_true, non_decision_true]) - np.array([drift_est, boundary_est, non_decision_est])
            biases.append(current_bias)
            squared_errors.append(current_bias**2)
    
    # Compute average bias and squared error over valid iterations.
    biases = np.array(biases)
    squared_errors = np.array(squared_errors)
    avg_bias = np.nanmean(biases, axis=0)
    avg_squared_error = np.nanmean(squared_errors, axis=0)
    valid_iterations = iterations - failed_count
    
    return avg_bias, avg_squared_error, valid_iterations, failed_count

def main():
    sample_sizes = [10, 40, 4000]
    
    for size in sample_sizes:
        avg_bias, avg_sq_error, valid_iters, invalid_iters = simulate_and_estimate(size)
        print(f"Sample size = {size}:")
        print("  Valid iterations:", valid_iters)
        print("  Invalid iterations:", invalid_iters)
        print("  Average Bias [drift, boundary, non_decision]:", avg_bias)
        print("  Average Squared Error [drift, boundary, non_decision]:", avg_sq_error)
        print("-----\n")

if __name__ == "__main__":
    main()
