import numpy as np

class EKF:
    def __init__(self, dynamics_func, measurement_func, state_dim, meas_dim, Q=None, R=None, x_init=None):
        """
        Extended Kalman Filter for a generic nonlinear system.

        Parameters:
        -----------
        dynamics_func : callable
            Function for the system dynamics: f(x, u, params).
        measurement_func : callable
            Function for the measurement model: h(x).
        state_dim : int
            Dimension of the state vector.
        meas_dim : int
            Dimension of the measurement vector.
        """
        self.f = dynamics_func        # System dynamics function
        self.h = measurement_func     # Measurement function
        self.state_dim = state_dim    # State vector dimension
        self.meas_dim = meas_dim      # Measurement vector dimension

        # Initialize state and covariance
        self.x = x_init if x_init is not None else np.zeros(state_dim)  # State estimate
        self.P = np.eye(state_dim)   # Covariance estimate

        # Process and measurement noise covariance
        self.Q = np.eye(state_dim) if Q == None else Q  # Process noise covariance
        self.R = np.eye(meas_dim) if R == None else R    # Measurement noise covariance

    def predict(self, u, dt, params={}):
        """
        Perform the prediction step of the EKF.

        Parameters:
        -----------
        u : np.array
            Control input vector.
        dt : float
            Time step.
        params : dict
            Additional parameters for the dynamics function.

        Updates:
        --------
        self.x : Predicted state vector.
        self.P : Predicted state covariance.
        """
        # Predict state using dynamics function
        self.x = self.x + self.f(self.x, u, params) * dt

        # Compute Jacobian of the dynamics function
        F = self.compute_jacobian(self.f, self.x, u, params)

        # Update state covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Perform the update step of the EKF.

        Parameters:
        -----------
        z : np.array
            Measurement vector.

        Updates:
        --------
        self.x : Updated state vector.
        self.P : Updated state covariance.
        """
        # Compute measurement prediction
        z_pred = self.h(self.x)

        # Compute Jacobian of the measurement function
        H = self.compute_jacobian(self.h, self.x)

        # Compute Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state estimate and covariance
        y = z - z_pred  # Measurement residual
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    @staticmethod
    def compute_jacobian(func, x, u=None, params={}):
        """
        Compute the Jacobian of a function using finite differences.

        Parameters:
        -----------
        func : callable
            The function for which the Jacobian is computed.
        x : np.array
            State vector.
        u : np.array (optional)
            Control input vector.
        params : dict (optional)
            Additional parameters for the function.

        Returns:
        --------
        np.array : The Jacobian matrix.
        """
        epsilon = 1e-5
        n = len(x)
        if u is None and isinstance(func(x), float):
            m = 1
        else:
            m = len(func(x, u, params)) if u is not None else len(func(x))
        J = np.zeros((m, n))
        for i in range(n):
            x_perturb = x.copy()
            x_perturb[i] += epsilon
            if u is not None:
                J[:, i] = (func(x_perturb, u, params) - func(x, u, params)) / epsilon
            else:
                J[:, i] = (func(x_perturb) - func(x)) / epsilon
        return J
    
    def get_state_estimate(self):
        """
        Returns the state estimate

        Returns:
        --------
        np.array : the state estimate
        """
        return self.x
    
    def get_state_estimate_covariance(self):
        """
        Returns the covariance of the state estimate

        Returns:
        --------
        np.array : the estimated covariance matrix of the state 
        """
        return self.P
