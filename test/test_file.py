#Code produced with Chat GPT assistance

#!/usr/bin/env python3

import unittest
import numpy as np

# Import functions and classes from your src modules.
from src.ez_diffusion import compute_forward_stats, simulate_summary_stats
from src.recovery import recover_parameters
from src.ez_diffusion_model import EZDiffusionModel


class TestEZDiffusion(unittest.TestCase):
    def setUp(self):
        self.a = 1.0       # boundary separation
        self.v = 1.0       # drift rate
        self.t = 0.3       # non-decision time
        self.tol = 1e-4

    def test_theoretical_calculations(self):
        R_pred, M_pred, V_pred = compute_forward_stats(self.a, self.v, self.t)
        y = np.exp(-self.a * self.v)
        expected_R = 1.0 / (1.0 + y)
        expected_M = self.t + (self.a / (2.0 * self.v)) * ((1.0 - y) / (1.0 + y))
        expected_V = (self.a / (2.0 * (self.v ** 3))) * ((1.0 - 2.0 * self.a * self.v * y - y**2) / ((1.0 + y)**2))
        self.assertTrue(np.isclose(R_pred, expected_R, atol=self.tol),
                        f"R_pred expected {expected_R}, got {R_pred}")
        self.assertTrue(np.isclose(M_pred, expected_M, atol=self.tol),
                        f"M_pred expected {expected_M}, got {M_pred}")
        self.assertTrue(np.isclose(V_pred, expected_V, atol=self.tol),
                        f"V_pred expected {expected_V}, got {V_pred}")

    def test_perfect_recovery_noise_free(self):
        R_pred, M_pred, V_pred = compute_forward_stats(self.a, self.v, self.t)
        a_est, v_est, t_est = recover_parameters(R_pred, M_pred, V_pred)
        self.assertTrue(np.isclose(a_est, self.a, atol=self.tol),
                        f"Recovered a {a_est} does not match {self.a}")
        self.assertTrue(np.isclose(v_est, self.v, atol=self.tol),
                        f"Recovered v {v_est} does not match {self.v}")
        self.assertTrue(np.isclose(t_est, self.t, atol=self.tol),
                        f"Recovered t {t_est} does not match {self.t}")

    def test_recovery_failure_unanimous(self):
        with self.assertRaises(ValueError):
            recover_parameters(1.0, 0.5, 0.02)
        with self.assertRaises(ValueError):
            recover_parameters(0.0, 0.5, 0.02)

    def test_numerical_stability_boundaries(self):
        R_pred, M_pred, V_pred = compute_forward_stats(self.a, 0.0, self.t)
        self.assertFalse(np.isnan(R_pred), "R_pred is NaN at zero drift")
        self.assertFalse(np.isnan(M_pred), "M_pred is NaN at zero drift")
        self.assertFalse(np.isnan(V_pred), "V_pred is NaN at zero drift")

    def test_parameter_recovery_simulated_noise(self):
        sample_sizes = [50, 100, 500]
        num_simulations = 100
        for N in sample_sizes:
            recovered_params = []
            for _ in range(num_simulations):
                sim_data = simulate_summary_stats(self.a, self.v, self.t, N)
                # Since simulate_summary_stats now returns a tuple,
                # we use tuple indexing: index 0 = R_obs, 1 = M_obs, 2 = V_obs.
                a_est, v_est, t_est = recover_parameters(sim_data[0], sim_data[1], sim_data[2])
                recovered_params.append((a_est, v_est, t_est))
            avg_a = np.mean([r[0] for r in recovered_params])
            avg_v = np.mean([r[1] for r in recovered_params])
            avg_t = np.mean([r[2] for r in recovered_params])
            self.assertTrue(np.isclose(avg_a, self.a, rtol=0.1),
                            f"Average recovered a {avg_a} not close to true {self.a}")
            self.assertTrue(np.isclose(avg_v, self.v, rtol=0.1),
                            f"Average recovered v {avg_v} not close to true {self.v}")
            self.assertTrue(np.isclose(avg_t, self.t, rtol=0.1),
                            f"Average recovered t {avg_t} not close to true {self.t}")

    def test_error_decreases_with_sample_size(self):
        N_small = 50
        N_large = 500
        num_simulations = 50
        errors_small = []
        errors_large = []
        for _ in range(num_simulations):
            sim_small = simulate_summary_stats(self.a, self.v, self.t, N_small)
            rec_small = recover_parameters(sim_small[0], sim_small[1], sim_small[2])
            error_small = abs(rec_small[0] - self.a) + abs(rec_small[1] - self.v) + abs(rec_small[2] - self.t)
            errors_small.append(error_small)

            sim_large = simulate_summary_stats(self.a, self.v, self.t, N_large)
            rec_large = recover_parameters(sim_large[0], sim_large[1], sim_large[2])
            error_large = abs(rec_large[0] - self.a) + abs(rec_large[1] - self.v) + abs(rec_large[2] - self.t)
            errors_large.append(error_large)

        mean_error_small = np.mean(errors_small)
        mean_error_large = np.mean(errors_large)
        self.assertTrue(mean_error_large < mean_error_small,
                        "Error did not decrease with larger sample size")

    def test_input_validation_invalid_parameters(self):
        with self.assertRaises(ValueError):
            compute_forward_stats(-1.0, self.v, self.t)
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, "not a number", self.t)
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, self.v, None)

    def test_challenging_numerical_scenarios(self):
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
        stats1 = compute_forward_stats(1.0, 1.0, 0.3)
        stats2 = compute_forward_stats(1.5, 1.5, 0.3)
        self.assertNotEqual(stats1, stats2,
                            "Different parameter sets produced identical forward stats")

    def test_parameter_sensitivity(self):
        stats_low_v = compute_forward_stats(self.a, 0.5, self.t)
        stats_high_v = compute_forward_stats(self.a, 1.5, self.t)
        self.assertTrue(stats_high_v[0] > stats_low_v[0],
                        "Increasing v did not increase R_pred")

        stats_low_a = compute_forward_stats(0.5, self.v, self.t)
        stats_high_a = compute_forward_stats(1.5, self.v, self.t)
        self.assertTrue(stats_high_a[1] > stats_low_a[1],
                        "Increasing a did not increase M_pred")
        self.assertTrue(stats_high_a[0] > stats_low_a[0],
                        "Increasing a did not increase R_pred for positive drift")

        stats_low_t = compute_forward_stats(self.a, self.v, 0.2)
        stats_high_t = compute_forward_stats(self.a, self.v, 0.4)
        self.assertTrue(np.isclose(stats_low_t[0], stats_high_t[0], atol=self.tol),
                        "Changing t unexpectedly changed R_pred")
        self.assertTrue(stats_high_t[1] > stats_low_t[1],
                        "Increasing t did not increase M_pred")

    def test_compute_forward_stats_non_numeric(self):
        with self.assertRaises(ValueError):
            compute_forward_stats("a", self.v, self.t)
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, "v", self.t)
        with self.assertRaises(ValueError):
            compute_forward_stats(self.a, self.v, "t")


class TestCorruption(unittest.TestCase):
    def setUp(self):
        # A valid simulated summary stats dictionary for the model.
        # Note: The EZDiffusionModel constructor accepts both a tuple or a dict.
        self.valid_data = {'R_obs': 0.8, 'M_obs': 0.5, 'V_obs': 0.02}
        self.a = 1.0
        self.v = 1.0
        self.t = 0.3
        self.model = EZDiffusionModel(data=self.valid_data, a=self.a, v=self.v, t=self.t)

    def test_invalid_data_constructor(self):
        with self.assertRaises(ValueError):
            EZDiffusionModel(data="not a tuple", a=self.a, v=self.v, t=self.t)
        with self.assertRaises(ValueError):
            EZDiffusionModel(data=(0.8,), a=self.a, v=self.v, t=self.t)
        with self.assertRaises(ValueError):
            EZDiffusionModel(data=(0.8, "invalid", 0.02), a=self.a, v=self.v, t=self.t)

    def test_recovered_parameters_read_only(self):
        with self.assertRaises(AttributeError):
            self.model.recovered_parameters = (2.0, 2.0, 0.4)

    def test_updating_data_recomputes_parameters(self):
        initial_params = self.model.recovered_parameters
        new_data = {'R_obs': 0.7, 'M_obs': 0.6, 'V_obs': 0.03}
        self.model.update_data(new_data)
        updated_params = self.model.recovered_parameters
        self.assertNotEqual(initial_params, updated_params,
                            "Recovered parameters did not change after updating data")


if __name__ == "__main__":
    unittest.main()
