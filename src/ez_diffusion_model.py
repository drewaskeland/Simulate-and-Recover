from src.recovery import recover_parameters

class EZDiffusionModel:
    def __init__(self, data, **kwargs):
        """
        Initialize the model with observed summary statistics.
        
        Parameters:
            data: Either a tuple (R_obs, M_obs, V_obs) or a dictionary with keys "R_obs", "M_obs", "V_obs".
            Extra keyword arguments (like a, v, t) are accepted but ignored.
        """
        if isinstance(data, dict):
            required_keys = ("R_obs", "M_obs", "V_obs")
            if not all(key in data for key in required_keys):
                raise ValueError("Data dictionary must have keys: (R_obs, M_obs, V_obs).")
            self._data = data
        elif isinstance(data, tuple):
            if len(data) != 3:
                raise ValueError("Data must be a tuple of three elements: (R_obs, M_obs, V_obs).")
            self._data = data
        else:
            raise ValueError("Data must be either a tuple or a dict with summary statistics.")
        self._compute_parameters()
    
    def _compute_parameters(self):
        try:
            if isinstance(self._data, dict):
                recovered = recover_parameters(self._data["R_obs"], self._data["M_obs"], self._data["V_obs"])
            else:
                recovered = recover_parameters(*self._data)
            self._a_est = recovered["a"]
            self._v_est = recovered["v"]
            self._t_est = recovered["t"]
        except Exception as e:
            raise ValueError("Parameter recovery failed: " + str(e))
    
    @property
    def data(self):
        """Return the observed summary statistics."""
        return self._data
    
    @data.setter
    def data(self, new_data):
        if isinstance(new_data, dict):
            required_keys = ("R_obs", "M_obs", "V_obs")
            if not all(key in new_data for key in required_keys):
                raise ValueError("Data dictionary must have keys: (R_obs, M_obs, V_obs).")
            self._data = new_data
        elif isinstance(new_data, tuple):
            if len(new_data) != 3:
                raise ValueError("Data must be a tuple of three elements: (R_obs, M_obs, V_obs).")
            self._data = new_data
        else:
            raise ValueError("Data must be either a tuple or a dict with summary statistics.")
        self._compute_parameters()
    
    @property
    def recovered_parameters(self):
        """Return the recovered parameters as a dictionary."""
        return {"a": self._a_est, "v": self._v_est, "t": self._t_est}
    
    def update_data(self, new_data):
        """Update the model's data and recompute recovered parameters."""
        self.data = new_data
    
    @property
    def a_est(self):
        """Recovered boundary separation."""
        return self._a_est
    
    @property
    def v_est(self):
        """Recovered drift rate."""
        return self._v_est
    
    @property
    def t_est(self):
        """Recovered nondecision time."""
        return self._t_est

if __name__ == "__main__":
    # Example observed summary statistics (R_obs, M_obs, V_obs)
    example_data = (0.75, 0.5, 0.04)
    model = EZDiffusionModel(example_data)
    print("Recovered drift rate:", model.v_est)
    print("Recovered boundary separation:", model.a_est)
    print("Recovered nondecision time:", model.t_est)