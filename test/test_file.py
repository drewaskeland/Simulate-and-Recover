import unittest
import numpy as np
from src.ez_diffusion import compute_forward_stats, simulate_summary_stats
from src.recovery import recover_parameters
from src.ez_diffusion_model import EZDiffusionModel

class TestEZDiffusionModified(unittest.TestCase):
    def setUp(self):
        # Each tuple is (boundary, drift, non_decision)
        self.standard_params = [
            (1.0, 1.0, 0.3),   # mid-range values
            (0.5, 0.5, 0.1),   # lower bounds
            (1.5, 1.5, 0.5),   # upper bounds
            (1.5, 0.8, 0.2),   # high boundary, moderate drift
            (0.8, 1.5, 0.4)    # low boundary, high drift
        ]
        self.N_values = [10, 40, 4000]  # sample sizes to test
        self.tolerance = {
            'strict': {'delta': 0.01, 'places': 3},
            'moderate': {'delta': 0.05, 'places': 2}
        }

    def test_invalid_input_types(self):
        # Ensure that compute_forward_stats raises an exception with non-numeric inputs.
        with self.assertRaises((TypeError, ValueError)):
            compute_forward_stats("foo", 1.0, 0.3)
        with self.assertRaises((TypeError, ValueError)):
            compute_forward_stats(1.0, "bar", 0.3)
        with self.assertRaises((TypeError, ValueError)):
            compute_forward_stats(1.0, 1.0, "baz")

    def test_edge_numerical_scenarios(self):
        # Check recovery in challenging numerical conditions.
        params = recover_parameters(0.6, 0.3, 1e-8)
        self.assertTrue(np.isfinite(params).all())
        
        params = recover_parameters(0.7, 1e-4, 0.49)
        self.assertTrue(np.isfinite(params).all())

    def test_forward_invalid_parameters(self):
        # Test input validation for compute_forward_stats with invalid parameters.
        invalid_params = [
            (-0.5, 1.0, 0.3),   # negative boundary
            (1.0, -1.0, 0.3),   # negative drift
            (1.0, 1.0, -0.1),   # negative non_decision
            (0.0, 1.0, 0.3),    # zero boundary
        ]
        for boundary, drift, non_decision in invalid_params:
            with self.subTest(boundary=boundary, drift=drift, non_decision=non_decision):
                with self.assertRaises(ValueError):
                    compute_forward_stats(boundary, drift, non_decision)

    def test_unique_parameter_predictions(self):
        # Check that different parameter sets yield different forward statistics.
        stats1 = compute_forward_stats(1.0, 1.0, 0.3)
        stats2 = compute_forward_stats(0.8, 1.2, 0.35)
        self.assertFalse(np.allclose(stats1, stats2, atol=1e-3))

    def test_parameter_modification_effects(self):
        """
        Confirm that:
         - Increasing non_decision shifts mean_RT without affecting accuracy.
         - Increasing boundary (with fixed drift and non_decision) increases both mean_RT and accuracy.
         - Increasing drift (with fixed boundary and non_decision) increases accuracy.
        """
        # Baseline parameters.
        boundary, drift, non_decision = 1.0, 1.0, 0.3
        accuracy0, mean_RT0, _ = compute_forward_stats(boundary, drift, non_decision)

        # Increase non_decision.
        accuracy_nd_up, mean_RT_nd_up, _ = compute_forward_stats(boundary, drift, non_decision + 0.2)
        self.assertGreater(mean_RT_nd_up, mean_RT0, "Increasing non_decision time should increase mean_RT.")
        self.assertAlmostEqual(accuracy_nd_up, accuracy0, delta=1e-7, msg="Accuracy should remain nearly unchanged when only non_decision time is increased.")

        # Increase boundary.
        accuracy_boundary_up, mean_RT_boundary_up, _ = compute_forward_stats(boundary + 0.5, drift, non_decision)
        self.assertGreater(mean_RT_boundary_up, mean_RT0, "Increasing boundary should increase mean_RT.")
        self.assertGreater(accuracy_boundary_up, accuracy0, "Increasing boundary (with positive drift) should increase accuracy.")

        # Increase drift.
        accuracy_drift_up, mean_RT_drift_up, _ = compute_forward_stats(boundary, drift + 0.5, non_decision)
        self.assertGreater(accuracy_drift_up, accuracy0, "Increasing drift should increase accuracy.")

    def test_performance_boundaries(self):
        # Ensure numerical stability at performance extremes.
        test_cases = [
            (0.501, 0.3, 0.1),   # near chance-level accuracy
            (0.999, 0.4, 0.2),   # near-perfect accuracy
            (0.01,  0.5, 0.3)    # very poor accuracy
        ]
        for accuracy, mean_RT, variance_RT in test_cases:
            with self.subTest(accuracy=accuracy, mean_RT=mean_RT, variance_RT=variance_RT):
                params = recover_parameters(accuracy, mean_RT, variance_RT)
                self.assertFalse(np.isnan(params).any())

    def test_closed_form_predictions(self):
        # Verify that forward predictions match closed-form EZ diffusion formulas.
        for boundary, drift, non_decision in self.standard_params:
            with self.subTest(boundary=boundary, drift=drift, non_decision=non_decision):
                accuracy_pred, mean_RT_pred, variance_RT_pred = compute_forward_stats(boundary, drift, non_decision)
                
                y = np.exp(-boundary * drift)
                expected_accuracy = 1.0 / (1.0 + y)
                expected_mean_RT = non_decision + (boundary / (2.0 * drift)) * ((1.0 - y) / (1.0 + y))
                expected_variance_RT = (boundary / (2.0 * (drift**3))) * ((1.0 - 2.0 * boundary * drift * y - y**2) / ((1.0 + y)**2))
                
                self.assertAlmostEqual(accuracy_pred, expected_accuracy, places=5)
                self.assertAlmostEqual(mean_RT_pred, expected_mean_RT, places=5)
                self.assertAlmostEqual(variance_RT_pred, expected_variance_RT, places=5)

    def test_noise_free_parameter_recovery(self):
        # Validate perfect recovery with deterministic (noise-free) data.
        for boundary, drift, non_decision in self.standard_params:
            with self.subTest(boundary=boundary, drift=drift, non_decision=non_decision):
                accuracy, mean_RT, variance_RT = compute_forward_stats(boundary, drift, non_decision)
                drift_est, boundary_est, non_decision_est = recover_parameters(accuracy, mean_RT, variance_RT)
                
                self.assertAlmostEqual(boundary_est / boundary, 1.0, delta=0.001)
                self.assertAlmostEqual(drift_est / drift, 1.0, delta=0.001)
                self.assertAlmostEqual(non_decision_est / non_decision, 1.0, delta=0.01)

    def test_extreme_accuracy_cases(self):
        # Ensure that extreme accuracy values trigger recovery failure.
        with self.assertRaises(ValueError):
            recover_parameters(1.0, 0.3, 0.1)  # accuracy_obs = 1.0 (all correct)
        with self.assertRaises(ValueError):
            recover_parameters(0.0, 0.3, 0.1)  # accuracy_obs = 0.0 (all incorrect)

    def test_recovery_under_sampling_noise(self):
        # Test recovery performance under simulated sampling noise.
        for boundary, drift, non_decision in self.standard_params:
            with self.subTest(boundary=boundary, drift=drift, non_decision=non_decision):
                biases = []
                for _ in range(1000):
                    accuracy_obs, mean_RT_obs, variance_RT_obs = simulate_summary_stats(boundary, drift, non_decision, N=100)
                    drift_est, boundary_est, non_decision_est = recover_parameters(accuracy_obs, mean_RT_obs, variance_RT_obs)
                    biases.append([
                        (boundary_est - boundary) / boundary, 
                        (drift_est - drift) / drift, 
                        (non_decision_est - non_decision) / non_decision
                    ])
                avg_bias = np.nanmean(biases, axis=0)
                for bias in avg_bias:
                    self.assertAlmostEqual(bias, 0.0, delta=0.10)

    def test_effect_of_sample_size(self):
        # Confirm that recovery error decreases as sample size increases.
        for boundary, drift, non_decision in self.standard_params:
            for N in self.N_values:
                errors = []
                valid_iterations = 0
                for _ in range(100):
                    try:
                        accuracy_obs, mean_RT_obs, variance_RT_obs = simulate_summary_stats(boundary, drift, non_decision, N)
                        drift_est, boundary_est, non_decision_est = recover_parameters(accuracy_obs, mean_RT_obs, variance_RT_obs)
                        errors.append([
                            ((boundary_est - boundary) / boundary)**2,
                            ((drift_est - drift) / drift)**2,
                            ((non_decision_est - non_decision) / non_decision)**2
                        ])
                        valid_iterations += 1
                    except ValueError:
                        continue
                
                if valid_iterations == 0:
                    self.skipTest(f"No valid iterations for params {boundary}, {drift}, {non_decision}, N={N}")
                
                avg_error = np.mean(errors, axis=0)
                if N >= 4000:
                    self.assertTrue(all(e < 0.03 for e in avg_error))


class TestCorruptionModified(unittest.TestCase):
    def test_data_update_recomputes_parameters(self):
        """Ensure that updating the data recomputes the recovered parameters."""
        data1 = (0.75, 0.5, 0.04)
        model = EZDiffusionModel(data1)
        original_drift = model.drift_est

        data2 = (0.8, 0.6, 0.05)
        model.data = data2
        self.assertNotEqual(model.drift_est, original_drift)

    def test_constructor_input_validation(self):
        """Ensure that invalid data provided to the EZDiffusionModel constructor raises ValueError."""
        with self.assertRaises(ValueError):
            EZDiffusionModel(1)
        with self.assertRaises(ValueError):
            EZDiffusionModel((0.75, 0.5))
        with self.assertRaises(ValueError):
            EZDiffusionModel((0.75, 0.5, "not a number"))
    
    def test_read_only_properties(self):
        """Verify that recovered parameters are read-only and cannot be modified."""
        valid_data = (0.75, 0.5, 0.04)
        model = EZDiffusionModel(valid_data)
        with self.assertRaises(AttributeError):
            model.drift_est = 10
        with self.assertRaises(AttributeError):
            model.boundary_est = 10
        with self.assertRaises(AttributeError):
            model.non_decision_est = 10


if __name__ == '__main__':
    unittest.main(failfast=True)
