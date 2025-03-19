#Code produced with Chat GPT assistance

from src.recovery import recover_parameters

class EZDiffusionModel:
    def __init__(self, data):
        """
        Create an instance of EZDiffusionModel using observed summary statistics.
        
        Parameters:
            data (tuple): A tuple containing (accuracy_obs, mean_RT_obs, variance_RT_obs).
        """
        if not (isinstance(data, tuple) and len(data) == 3):
            raise ValueError("Data must be a tuple with three elements: (accuracy_obs, mean_RT_obs, variance_RT_obs).")
        self._data = data
        self._compute_parameters()
    
    def _compute_parameters(self):
        try:
            # Call recover_parameters, which is expected to yield (drift_est, boundary_est, non_decision_est)
            self._drift_est, self._boundary_est, self._non_decision_est = recover_parameters(*self._data)
        except Exception as err:
            raise ValueError("Parameter recovery failed: " + str(err))
    
    @property
    def data(self):
        """Retrieve the current observed summary statistics."""
        return self._data
    
    @data.setter
    def data(self, new_data):
        if not (isinstance(new_data, tuple) and len(new_data) == 3):
            raise ValueError("Data must be a tuple with three elements: (accuracy_obs, mean_RT_obs, variance_RT_obs).")
        self._data = new_data
        self._compute_parameters()  # Recalculate the recovered parameters with the new data immediately
    
    @property
    def drift_est(self):
        """Return the recovered drift rate."""
        return self._drift_est
    
    @property
    def boundary_est(self):
        """Return the recovered boundary separation."""
        return self._boundary_est
    
    @property
    def non_decision_est(self):
        """Return the recovered non-decision time."""
        return self._non_decision_est


if __name__ == "__main__":
    # Sample observed summary statistics: (accuracy_obs, mean_RT_obs, variance_RT_obs)
    sample_data = (0.75, 0.5, 0.04)
    model = EZDiffusionModel(sample_data)
    print("Recovered drift rate:", model.drift_est)
    print("Recovered boundary separation:", model.boundary_est)
    print("Recovered non-decision time:", model.non_decision_est)
