import numpy as np
import matplotlib.pyplot as plt

from pid_controller import PIDController
from ekf import EKF
from anomaly_detection import CUSUMDetector, ChiSquaredDetector
from metrics import MetricsTracker, plot_figs
from .base_control_system import BaseControlSystem

class HillClimbingCar(BaseControlSystem):

    def __init__(self):
        super().__init__()

    def dynamics(self, state, inputs, params):
        """
        Vehicle dynamics with a road slope.

        Parameters:
            state: np.array [x, v]
                - x: Position (m)
                - v: Velocity (m/s)
            inputs: np.array [throttle]
                - throttle: Float between 0 and 1
            params: dict
                - m: Mass of the vehicle (kg)
                - g: Gravitational acceleration (m/s^2)
                - Cr: Coefficient of rolling resistance
                - Cd: Aerodynamic drag coefficient
                - A: Frontal area (m^2)
                - rho: Air density (kg/m^3)
                - Fmax: Maximum engine force (N)

        Returns:
            np.array: Derivatives [dx/dt, dv/dt]
        """
        x, v = state
        throttle = inputs[0]
        
        # Extract parameters
        m = params.get("m", 1200)       # Vehicle mass (kg)
        g = params.get("g", 9.81)       # Gravity (m/s^2)
        Cr = params.get("Cr", 0.01)     # Rolling resistance coefficient
        Cd = params.get("Cd", 0.3)      # Aerodynamic drag coefficient
        A = params.get("A", 2.2)        # Frontal area (m^2)
        rho = params.get("rho", 1.225)  # Air density (kg/m^3)
        Fmax = params.get("Fmax", 4000) # Maximum engine force (N)
        
        # Compute the road slope
        theta = self.road_slope(x)
        
        # Forces
        F_engine = throttle * Fmax                          # Engine force
        F_gravity = m * g * np.sin(theta)                   # Gravity force along the slope
        F_rolling = m * g * Cr * np.cos(theta)              # Rolling resistance
        F_drag = 0.5 * rho * Cd * A * v**2 *np.cos(theta)   # Aerodynamic drag
        
        # Acceleration
        F_total = F_engine - (F_gravity + F_rolling + F_drag)
        a = F_total / m  # Newton's second law
        
        return np.array([v, a])
    
    def measurement_func(self, state):
        """
        Measurement function, mepping states to measurements
        We can only measure the velocity in this setup

        Parameters:
            state: np.array [x, v]
                - x: Position (m)
                - v: Velocity (m/s)
        Returns:
            float: Velocity of car
        """
        pos, vel = state
        return vel
    
    def road_slope(self, x):
        """
        Define the road slope as a function of position.
        
        Parameters:
            x: float
                Position along the road (m)
        
        Returns:
            float: Slope angle (in radians)
        """
        # return 0.1 * np.sin(0.05 * x)  # 10% grade sinusoidal hill
        if x >= 10 and x < 300:
            return np.pi / 6
        else: 
            return 0.0
        
    def attack(self, measurement, magnitude=0.3):
        """
        Attack the measurement signal with some magnitude
        TODO: Implement more types of attacks

        Parameters:
            measurement: float
            magnitude: float
        Returns:
            float
        """
        attacked_measurement = measurement + magnitude
        return attacked_measurement
    
    def run_control_system(self, config=
        {
            "params": {
                "m": 1200,       # Mass (kg)
                "g": 9.81,       # Gravity (m/s^2)
                "Cr": 0.01,      # Rolling resistance coefficient
                "Cd": 0.3,       # Aerodynamic drag coefficient
                "A": 2.2,        # Frontal area (m^2)
                "rho": 1.225,    # Air density (kg/m^3)
                "Fmax": 40000,    # Maximum engine force (N)
            },
            "init-state": np.array([0, 5]),  # [Position (m), Velocity (m/s)]
            "dt": 0.1,
            "time": 100,
            "attack-start": 10,
            "attack-end": 20,
            "attack-magnitude": 7.0,
            "v-controller": PIDController(0.50320036, 0.50027134, 0.00711447),
            "target-velocity": 20,
            "process-noise-cov": np.diag([0.01, 0.1]),
            "measurement-noise-cov": 0.2,
            "anomaly-detector": CUSUMDetector(
                thresholds=np.array([4.8]),
                b=np.array([4.79])),
        },
        show_plots=False
    ):
        # Process and measurement noise
        process_noise_cov = config['process-noise-cov']
        measurement_noise_cov = config['measurement-noise-cov']
        
        # Initial state and parameters
        state = config['init-state']  # [Position (m), Velocity (m/s)]
        measurement = self.measurement_func(state) + np.random.normal(0, measurement_noise_cov)
        params = config['params']
        throttle = 0

        dt = config['dt']        # Time step
        time = np.arange(0, config['time'], dt)
        attack_start = config['attack-start']
        attack_end = config['attack-end']

        # Controller init and target velocity
        v_controller: PIDController = config['v-controller']
        target_velocity = config['target-velocity']

        # Init Extended Kalman Filter
        ekf = EKF(
            dynamics_func=self.dynamics,
            measurement_func=self.measurement_func,
            state_dim=2,
            meas_dim=1,
            x_init=state
        )

        detection_ekf = EKF(
            dynamics_func=self.dynamics,
            measurement_func=self.measurement_func,
            state_dim=2,
            meas_dim=1,
            x_init=state
        )

        # Anomaly Detector
        cusum: CUSUMDetector = config['anomaly-detector']
        chi2 = ChiSquaredDetector()
        under_attack = False
        # Save metrics
        tracker = MetricsTracker()

        place_holder_cond = False
        integral_vals = []

        # Simulate
        for t in time:

            # if t > 30:
            #     breakpoint()

            # Predict state
            estimated_state = detection_ekf.get_state_estimate()
            estimated_speed_variance = detection_ekf.get_state_estimate_covariance()[1,1]
            
            # Take measurement
            measurement = self.measurement_func(state) + np.random.normal(0, measurement_noise_cov)
            # Calculate residual and send to anomaly detector
            residual = measurement - self.measurement_func(estimated_state)
            _, under_attack = cusum.update(residual)
            # under_attack = chi2.update(residual, estimated_speed_variance)
            # Launch Attack
            if t > attack_start and t < attack_end:
                measurement = self.attack(measurement, magnitude=config['attack-magnitude'])
            # Derive control input
            throttle = v_controller.compute(target_velocity, measurement, dt)
            integral_vals.append(v_controller.get_integral())
            if under_attack:
                # Compute control based on estimated state
                # breakpoint()
                estimated_state = ekf.get_state_estimate()
                est_measurement = self.measurement_func(estimated_state)
                throttle = v_controller.compute(target_velocity, est_measurement, dt)
                # Prediction step but no correction step due to attacked sensor measurement
                ekf.predict([throttle], dt, params)
            else:
                # Prediction and Correction if no attack
                ekf.predict([throttle], dt, params)
                ekf.update(np.array([measurement]))
            
            # Always Prediction and Correction for Detection EKF
            detection_ekf.predict([throttle], dt, params)
            detection_ekf.update(np.array([measurement]))

            # Compute next state
            derivatives = self.dynamics(state, [throttle], params) 
            tracker.track(state, estimated_state, measurement, throttle, residual, under_attack)
            state = state + derivatives * dt + np.random.multivariate_normal(np.zeros(2), process_noise_cov)
            
        # print(tracker.residual_statistics())
        # print(f"Mean Squared Control Error: {np.mean(tracker.ms_control_error())}")

        if show_plots:
            plot_figs(time, tracker)
            tracker.plot_attack_predictions()
            # Plot the road slope as a function of position
            plt.figure(figsize=(6, 4))
            states = np.array(tracker.get_metrics()[0])
            positions = states[:, 0]
            slopes = np.array([self.road_slope(pos) for pos in positions])
            plt.plot(positions, slopes, label="Road Slope (rad)")
            plt.xlabel("Position (m)")
            plt.ylabel("Slope (rad)")
            plt.title("Road Slope Profile")
            plt.legend()

            plt.figure()
            plt.plot([x/10 for x in range(len(integral_vals))], integral_vals)
            plt.title("Integral in Controller over Time")

            plt.show()

        return tracker
