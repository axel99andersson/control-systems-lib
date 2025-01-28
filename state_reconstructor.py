from collections import deque

from ekf import EKF

class CheckPointer:
    def __init__(self, detection_delay):
        self.queue = deque(maxlen=detection_delay)

    def update_checkpoint(self, mean, var, control_signal):
        """
        Adds a state estimate and control signal to the queue
        --------------
        Args: 
            mean (np.array): mean of the state estimate
            var (np.array): variance of the state estimate
            control_signal (np.array): the control signal
        """
        self.queue.append((mean, var, control_signal))

    def get_trusty_state(self):
        """
        Retrieves the last recorded trustful state and all control
        signals used after this state
        ------------
        Returns:
            Tuple[np.array, np.array, List[np.array]]: the mean and variance
                of the last trustful state and the sequence of ctr signals
                used after
        """
        mean, var, _ = self.queue[0]
        ctr_signals = [x[2] for x in self.queue]
        self.queue.clear()
        return mean, var, ctr_signals

    
class StateReconstructor:
    def __init__(self, dt, dynamics_func, measurement_func, state_dim, meas_dim, Q=None, R=None, x_init=None):
        self.ekf = EKF(dynamics_func, measurement_func, state_dim, meas_dim, Q, R, x_init)
        self.dt = dt

    def reconstruct_state(self, checkpointer: CheckPointer):
        """
        Reconstructs the current state based on the CheckPointer

        Args:
            checkpointer : CheckPointer
        
        Returns:
            Tuple[np.array, np.array] :
                the reconstructed state estimate and variance
        """
        state_mean, state_var, ctr_signals = checkpointer.get_trusty_state()
        self.ekf.set_state_estimate(state_mean, state_var)

        for u in ctr_signals:
            self.ekf.predict(u, self.dt)

        return self.ekf.get_state_estimate(), self.ekf.get_state_estimate_covariance()

