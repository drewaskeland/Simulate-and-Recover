#Code produced with Chat GPT assistance

import numpy as np
from ez_diffusion import simulate_summary_stats
from recovery import recover_parameters

def simulate_and_recover(a_true, v_true, t_true, N, iterations=1000):
    """
    Perform the simulate-and-recover loop for a given sample size N for a fixed number
    of iterations. All iterations are recorded; if an iteration yields invalid recovered
    parameters, it is recorded as NaN.
    
    Parameters:
        a_true (float): True boundary separation.
        v_true (float): True drift rate.
        t_true (float): True nondecision time.
        N (int): Sample size (number of trials).
        iterations (int): Total number of iterations to run.
    
    Returns:
        avg_bias (ndarray): Average bias for [a, v, t] computed over valid iterations.
        avg_squared_error (ndarray): Average squared error for [a, v, t] computed over valid iterations.
        valid_iterations (int): Number of iterations with valid recovered parameters.
        invalid_iterations (int): Number of iterations with invalid recovered parameters.
    """
    biases = []
    squared_errors = []
    invalid_count = 0
    
    for _ in range(iterations):
        # Simulate summary statistics for given true parameters and sample size.
        R_obs, M_obs, V_obs = simulate_summary_stats(a_true, v_true, t_true, N)
        
        # Recover parameters (returns a tuple: (a_est, v_est, t_est)).
        a_est, v_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
        
        # Check if recovered parameters are valid (i.e. not NaN)
        if np.isnan(a_est) or np.isnan(v_est) or np.isnan(t_est):
            biases.append(np.array([np.nan, np.nan, np.nan]))
            squared_errors.append(np.array([np.nan, np.nan, np.nan]))
            invalid_count += 1
        else:
            # Compute bias using the order: [a, v, t]
            bias = np.array([a_true, v_true, t_true]) - np.array([a_est, v_est, t_est])
            biases.append(bias)
            squared_errors.append(bias**2)
    
    biases = np.array(biases)
    squared_errors = np.array(squared_errors)
    
    # Use nanmean to average only over valid iterations.
    avg_bias = np.nanmean(biases, axis=0)
    avg_squared_error = np.nanmean(squared_errors, axis=0)
    valid_iterations = iterations - invalid_count
    
    return avg_bias, avg_squared_error, valid_iterations, invalid_count

def main():
    a_true = 1.0   # True boundary separation
    v_true = 1.0   # True drift rate
    t_true = 0.3   # True nondecision time
    sample_sizes = [10, 40, 4000]
    
    for N in sample_sizes:
        avg_bias, avg_squared_error, valid_iters, invalid_iters = simulate_and_recover(a_true, v_true, t_true, N)
        print(f"Sample size N = {N}:")
        print("Valid iterations:", valid_iters)
        print("Invalid iterations:", invalid_iters)
        print("Average Bias [a, v, t]:", avg_bias)
        print("Average Squared Error [a, v, t]:", avg_squared_error)
        print("-----\n")

if __name__ == "__main__":
    main()
