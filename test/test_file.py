#!/usr/bin/env python3

import unittest
import numpy as np

# Import functions and classes from your src modules.
from src.ez_diffusion import compute_forward_stats, simulate_summary_stats
from src.recovery import recover_parameters
from src.ez_diffusion_model import EZDiffusionModel


class TestEZDiffusion(unittest.TestCase):
    def setUp(self):
        # Set up common parameters and tolerance for floating point comparisons.
        self.a = 1.0       # boundary separation
        self.v = 1.0       # drift rate
        self.t = 0.3       # non-decision time
        self.tol = 1e-4

    def test_theoretical_calculations(self):
        """
        Test theoretical calculations against the standard closed‐form solutions.
        (Replace the dummy expected values with the actual theoretical values from Week 9 slides.)
        """
        R_pred, M_pred, V_pred = compute_forward_stats(self.a, self.v, self.t)
        expected_R = 0.8   # dummy expected value; change to the correct one
        expected_M = 0.5   # dummy expected value; change to the correct one
        expected_V = 0.02  # dummy expected value; change to the correct one
        self.assertTrue(np.isclose(R_pred, expected_R, atol=self.tol),
                        f"R_pred expected {expected_R}, got {R_pred}")
        self.assertTrue(np.isclose(M_pred, expected_M, atol=self.tol),
                        f"M_pred expected {expected_M}, got {M_pred}")
        self.assertTrue(np.isclose(V_pred, expected_V, atol=self.tol),
                        f"V_pred expected {expected_V}, got {V_pred}")

    def test_perfect_recovery_noise_free(self):
        """
        Test that noise-free forward predictions exactly recover the original parameters.
        """
        R_pred, M_pred, V_pred = compute_forward_stats(self.a, self.v, self.t)
        recovered = recover_parameters(R_pred, M_pred, V_pred)
        self.assertTrue(np.isclose(recovered['a'], self.a, atol=self.tol),
                        f"Recovered a {recovered['a']} does not match {self.a}")
        self.assertTrue(np.isclose(recovered['v'], self.v, atol=self.tol),
                        f"Recovered v {recovered['v']} does not match {self.v}")
        self.assertTrue(np.isclose(recovered['t'], self.t, atol=self.tol),
                        f"Recovered t {recovered['t']} does not match {self.t}")

    def test_recovery_failure_unanimous(self):
        """
        Test that when the observed accuracy is 1.0 or 0.0, the recovery function raises a ValueError.
        """
        with self.assertRaises(ValueError):
            recover_parameters(1.0, 0.5, 0.02)
        with self.assertRaises(ValueError):
            recover_parameters(0.0, 0.5, 0.02)

    def test_numerical_stability_boundaries(self):
        """
        Test that forward statistics remain numerically stable at performance boundaries.
        """
        R_pred, M_pred, V_pred = compute_forward_stats(self.a, 0.0, self.t)
        self.assertFalse(np.isnan(R_pred), "R_pred is NaN at zero drift")
        self.assertFalse(np.isnan(M_pred), "M_pred is NaN at zero drift")
        self.assertFalse(np.isnan(V_pred), "V_pred is NaN at zero drift")

    def test_parameter_recovery_simulated_noise(self):
        """
        With simulated (noisy) data, verify that on average the recovered parameters are close to the true parameters.
        """
        sample_sizes = [50, 100, 500]
        num_simulations = 100  # moderate count for testing
        for N in sample_sizes:
            recovered_params = []
            for _ in range(num_simulations):
                sim_data = simulate_summary_stats(self.a, self.v, self.t, N)
                rec = recover_parameters(sim_data['R_obs'], sim_data['M_obs'], sim_data['V_obs'])
                recovered_params.append(rec)
            avg_a = np.mean([r['a'] for r in recovered_params])
            avg_v = np.mean([r['v'] for r in recovered_params])
            avg_t = np.mean([r['t'] for r in recovered_params])
            self.assertTrue(np.isclose(avg_a, self.a, rtol=0.1),
                            f"Average recovered a {avg_a} not close to true {self.a}")
            self.assertTrue(np.isclose(avg_v, self.v, rtol=0.1),
                            f"Average recovered v {avg_v} not close to true {self.v}")
            self.assertTrue(np.isclose(avg_t, self.t, rtol=0.1),
                            f"Average recovered t {avg_t} not close to true {self.t}")

    def test_error_decreases_with_sample_size(self):
        """
        Test that error typically decreases with increasing sample size.
        """
        N_small = 50
        N_large = 500
        num_simulations = 50
        errors_small = []
        errors_large = []
        for _ in range(num_simulations):
            sim_small = simulate_summary_stats(self.a, self.v, self.t, N_small)
            rec_small = recover_parameters(sim_small['R_obs'], sim_small['M_obs'], sim_small['V_obs'])
            error_small = abs(rec_small['a'] - self.a) + abs(rec_small['v'] - self.v) + abs(rec_small['t'] - self.t)
            errors_small.append(error_small)

            sim_large = simulate_summary_stats(self.a, self.v, self.t, N_large)
            rec_large = recover_parameters(sim_large['R_obs'], sim_large['M_obs'], sim_large['V_obs'])
            error_large = abs(rec_large['a'] - self.a) + abs(rec_large['v'] - self.v) + abs(rec_large['t'] - self.t)
            errors_large.append(error_large)

        mean_error_small = np.mean(errors_small)
        mean_error_large = np.mean(errors_large)
        self.assertTrue(mean_error_large < mean_error_small,
                        "Error did not decrease with larger sample size")

    def test_input_validation_invalid_parameters(self):
        """
        Test that compute_forward_stats raises ValueError for invalid inputs.
        """
        with self.assertRaises(ValueError):
            compute_forward_stats(-1.0, self.v, self.t)  # negative a not allowed
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, "not a number", self.t)
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, self.v, None)

    def test_challenging_numerical_scenarios(self):
        """
        Test challenging numerical scenarios to ensure outputs remain within valid ranges.
        """
        test_cases = [
            (1.0, 0.01, 0.3),
            (5.0, 2.0, 0.2),
            (0.5, 0.5, 0.5)
        ]
        for a_val, v_val, t_val in test_cases:
            R_pred, M_pred, V_pred = compute_forward_stats(a_val, v_val, t_val)
            self.assertTrue(0 <= R_pred <= 1, "R_pred out of bounds")
            self.assertTrue(M_pred > 0, "M_pred must be > 0")
            self.assertTrue(V_pred > 0, "V_pred must be > 0")

    def test_distinct_parameter_sets_different_forward_stats(self):
        """
        Test that distinct parameter sets produce different forward statistics.
        """
        stats1 = compute_forward_stats(1.0, 1.0, 0.3)
        stats2 = compute_forward_stats(1.5, 1.5, 0.3)
        self.assertNotEqual(stats1, stats2,
                            "Different parameter sets produced identical forward stats")

    def test_parameter_sensitivity(self):
        """
        Check that:
          - Increasing v increases R_pred.
          - Increasing a increases both M_pred and R_pred (for positive drift).
          - Increasing t primarily shifts M_pred without changing R_pred.
        """
        # Vary drift rate v.
        stats_low_v = compute_forward_stats(self.a, 0.5, self.t)
        stats_high_v = compute_forward_stats(self.a, 1.5, self.t)
        self.assertTrue(stats_high_v[0] > stats_low_v[0],
                        "Increasing v did not increase R_pred")

        # Vary boundary separation a.
        stats_low_a = compute_forward_stats(0.5, self.v, self.t)
        stats_high_a = compute_forward_stats(1.5, self.v, self.t)
        self.assertTrue(stats_high_a[1] > stats_low_a[1],
                        "Increasing a did not increase M_pred")
        self.assertTrue(stats_high_a[0] > stats_low_a[0],
                        "Increasing a did not increase R_pred for positive drift")

        # Vary non-decision time t.
        stats_low_t = compute_forward_stats(self.a, self.v, 0.2)
        stats_high_t = compute_forward_stats(self.a, self.v, 0.4)
        self.assertTrue(np.isclose(stats_low_t[0], stats_high_t[0], atol=self.tol),
                        "Changing t unexpectedly changed R_pred")
        self.assertTrue(stats_high_t[1] > stats_low_t[1],
                        "Increasing t did not increase M_pred")

    def test_compute_forward_stats_non_numeric(self):
        """
        Test that compute_forward_stats raises a ValueError when passed non-numeric values.
        """
        with self.assertRaises(ValueError):
            compute_forward_stats("a", self.v, self.t)
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, "v", self.t)
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, self.v, "t")


class TestCorruption(unittest.TestCase):
    def setUp(self):
        # A valid simulated summary stats dictionary for the model.
        self.valid_data = {'R_obs': 0.8, 'M_obs': 0.5, 'V_obs': 0.02}
        self.a = 1.0
        self.v = 1.0
        self.t = 0.3
        self.model = EZDiffusionModel(data=self.valid_data, a=self.a, v=self.v, t=self.t)

    def test_invalid_data_constructor(self):
        """
        Test that providing invalid data to the EZDiffusionModel constructor raises ValueError.
        The invalid inputs include:
          - A value that is not a tuple.
          - A tuple with too few elements.
          - A tuple with non-numeric element.
        """
        with self.assertRaises(ValueError):
            EZDiffusionModel(data="not a tuple", a=self.a, v=self.v, t=self.t)
        with self.assertRaises(ValueError):
            EZDiffusionModel(data=(0.8,), a=self.a, v=self.v, t=self.t)  # too few elements
        with self.assertRaises(ValueError):
            EZDiffusionModel(data=(0.8, "invalid", 0.02), a=self.a, v=self.v, t=self.t)

    def test_recovered_parameters_read_only(self):
        """
        Test that the recovered_parameters property is read-only and cannot be set.
        """
        with self.assertRaises(AttributeError):
            self.model.recovered_parameters = {'a': 2.0, 'v': 2.0, 't': 0.4}

    def test_updating_data_recomputes_parameters(self):
        """
        Test that updating the model's data recomputes the recovered_parameters.
        """
        initial_params = self.model.recovered_parameters
        new_data = {'R_obs': 0.7, 'M_obs': 0.6, 'V_obs': 0.03}
        self.model.update_data(new_data)
        updated_params = self.model.recovered_parameters
        self.assertNotEqual(initial_params, updated_params,
                            "Recovered parameters did not change after updating data")


if __name__ == "__main__":
    unittest.main()
