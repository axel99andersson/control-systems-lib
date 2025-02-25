import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import json

from pid_controller import PIDController
from recovery_controllers import LSTMRecoveryController

from ekf import EKF
from anomaly_detection import CUSUMDetector, ChiSquaredDetector, LikelihoodDetector
from state_reconstructor import CheckPointer, StateReconstructor
from metrics import MetricsTracker, Logger, plot_figs
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
        if x >= 100 and x < 300:
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
                "Fmax": 7000,    # Maximum engine force (N) 40000
            },
            "init-state": np.array([0.0, 19.0]),  # [Position (m), Velocity (m/s)]
            "dt": 0.01,
            "time": 50,
            "attack-start": -10,
            "attack-end": -20,
            "attack-magnitude": 5.0,
            "v-controller": PIDController(1.0, 1.0, 0.0), # PIDController(0.1, 0.5, 0), 
            "target-velocity": 20,
            "process-noise-cov": np.diag([0.001, 0.001]),
            "measurement-noise-cov": 0.1,
            "anomaly-detector": CUSUMDetector(
                thresholds=np.array([5.36377434494803]), # 4.0 Before
                b=np.array([0.5321249776851772])), # 1.7 Before
            "mission": [
                # Missions here... e.g (10.0, 13.5) = at time = 10.0, change to setpoint 13.5
            ]
        },
        seed=0,
        recovery_module_on=True,
        show_plots=False,
        save_data=True,
    ):  
        
        # Set random Seed
        np.random.seed(seed)

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

        mission = config['mission']
        mission_times = [x[0] for x in mission]

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

        detection_ekf_lagged = EKF(
            dynamics_func=self.dynamics,
            measurement_func=self.measurement_func,
            state_dim=2,
            meas_dim=1,
            x_init=state
        )

        reconstructor = StateReconstructor(dt=dt)
        checkpointer = CheckPointer(detection_delay=50)

        # Anomaly Detector
        cusum: CUSUMDetector = config['anomaly-detector']
        chi2 = ChiSquaredDetector()
        ld = LikelihoodDetector(threshold=0.995)

        under_attack = False
        prev_under_attack = False
        # Save metrics
        tracker = MetricsTracker()
        logger = Logger()
        
        # Simulate
        for t in time:
            
            # Change setpoint based on defined mission
            try:
                mission_index = mission_times.index(round(t, 1))
                target_velocity = mission[mission_index][1]
            except:
                pass
                
            # Predict state
            estimated_state = detection_ekf.get_state_estimate()
            estimated_state_variance = detection_ekf.get_state_estimate_covariance()
            
            # Take measurement
            measurement = self.measurement_func(state) + np.random.normal(measurement_noise_cov)
            
            # Launch Attack at some timesteps
            if t > attack_start and t < attack_end:
                measurement = self.attack(measurement, magnitude=config['attack-magnitude'])
            
            # Update the measurement tracking EKF
            detection_ekf.update(np.array([measurement]))
            estimated_state_after_correction = detection_ekf.get_state_estimate()

            # Calculate residual and send to anomaly detector
            residual = self.measurement_func(detection_ekf.get_state_estimate()) - \
                        self.measurement_func(detection_ekf_lagged.get_state_estimate()) # self.measurement_func(estimated_state) - ...
            _, change_detected = cusum.update(residual)
            
            if change_detected:
                under_attack = not under_attack
                cusum.reset()
            # under_attack = ld.update(residual)
            # under_attack = chi2.update(residual, 1.1667472993160948)
            # Derive control input
            throttle = v_controller.compute(target_velocity, estimated_state_after_correction[1], dt)
            
            # In case the recovery module is disengaged
            if not recovery_module_on:
                under_attack = False
            
            if under_attack:
                
                if not prev_under_attack:
                    estimated_state, _ = reconstructor.reconstruct_state(checkpointer, ekf)
                else:
                    # Compute control based on estimated state
                    estimated_state = ekf.get_state_estimate()
                    estimated_state_variance = ekf.get_state_estimate_covariance()
                est_measurement = self.measurement_func(estimated_state)
                throttle = v_controller.compute(target_velocity, est_measurement, dt)
                # Prediction step but no correction step due to attacked sensor measurement
                ekf.predict([throttle], dt, params)
                
            else:
                # Prediction and Correction if no attack
                ekf.predict([throttle], dt, params)
                ekf.update(np.array([measurement]))
            
            # Update Checkpointer
            checkpointer.update_checkpoint(estimated_state, estimated_state_variance, [throttle])
            
            # Always Prediction and Correction for Detection EKF
            detection_ekf.predict([throttle], dt, params)
            detection_ekf_lagged.predict([throttle], dt, params)
            detection_ekf_lagged.update(np.array([measurement]))
            # detection_ekf.update(np.array([measurement]))

            # Compute next state
            derivatives = self.dynamics(state, [throttle], params) 
            tracker.track(state, estimated_state, measurement, throttle, residual, under_attack)
            # Log data
            logger.log_data({
                "time": t,
                "pos": state[0],
                "vel": state[1],
                "est_pos": estimated_state[0],
                "est_vel": estimated_state[1],
                "det_est_pos": estimated_state_after_correction[0],
                "det_est_vel": estimated_state_after_correction[1],
                "measured_vel": measurement,
                "reference_vel": target_velocity,
                "ctl_signal": throttle,
                "attack": attack_start < t < attack_end,
                "attack_pred": under_attack,
                "attack_mag": config['attack-magnitude'] if attack_start < t < attack_end else 0.0,
                "residual": residual,
                "cusum_stat": cusum.get_cusum_statistic()
            })
            state = state + derivatives * dt + np.random.multivariate_normal(np.zeros(2), process_noise_cov)
            prev_under_attack = under_attack


        # print(tracker.residual_statistics())
        # print(f"Mean Squared Control Error: {np.mean(tracker.ms_control_error())}")
        if save_data:
            logger.save_data("car")
        if show_plots:
            plot_figs(time, tracker)
            # tracker.plot_attack_predictions()
            # # Plot the road slope as a function of position
            plt.figure(figsize=(6, 4))
            positions = np.arange(0, 1000, dt)
            slopes = np.array([self.road_slope(pos) for pos in positions])
            plt.plot(positions, slopes, label="Road Slope (rad)")
            plt.xlabel("Position (m)")
            plt.ylabel("Slope (rad)")
            plt.title("Road Slope Profile")
            plt.legend()

            plt.show()

        return logger

    def run_control_system_torch(self, config=
        {
            "params": {
                "m": 1200,       # Mass (kg)
                "g": 9.81,       # Gravity (m/s^2)
                "Cr": 0.01,      # Rolling resistance coefficient
                "Cd": 0.3,       # Aerodynamic drag coefficient
                "A": 2.2,        # Frontal area (m^2)
                "rho": 1.225,    # Air density (kg/m^3)
                "Fmax": 7000,    # Maximum engine force (N) 40000
            },
            "init-state": np.array([0.0, 19.0]),  # [Position (m), Velocity (m/s)]
            "dt": 0.01,
            "time": 50,
            "attack-start": 10,
            "attack-end": 20,
            "attack-magnitude": 5.0,
            "v-controller": PIDController(1.0, 1.0, 0.0), # PIDController(0.1, 0.5, 0), 
            "target-velocity": 20,
            "process-noise-cov": np.diag([0.001, 0.001]),
            "measurement-noise-cov": 0.1,
            "anomaly-detector": CUSUMDetector(
                thresholds=np.array([5.36377434494803]), # 4.0 Before
                b=np.array([0.5321249776851772])), # 1.7 Before
            "mission": [
                # Missions here... e.g (10.0, 13.5) = at time = 10.0, change to setpoint 13.5
            ]
        },
        recovery_module_on=True,
        show_plots=False,
        save_data=True
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

        mission = config['mission']
        mission_times = [x[0] for x in mission]

        # Controller init and target velocity
        v_controller = LSTMRecoveryController(input_size=5, hidden_size=8, dropout=0.2)
        v_controller.load_state_dict(torch.load("./models/controller_weights_ep46.pth", map_location=torch.device('cpu'), weights_only=True))
        v_controller.eval()
        with open("./models/scaler.json", "r") as f:
            loaded_params = json.load(f)

        # Create a new scaler and set its attributes
        scaler = MinMaxScaler(**loaded_params["params"])
        scaler.data_min_ = np.array(loaded_params["data_min"])
        scaler.data_max_ = np.array(loaded_params["data_max"])
        scaler.data_range_ = np.array(loaded_params["data_range"])
        scaler.scale_ = np.array(loaded_params["scale"])
        scaler.min_ = np.array(loaded_params["min"])

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

        detection_ekf_lagged = EKF(
            dynamics_func=self.dynamics,
            measurement_func=self.measurement_func,
            state_dim=2,
            meas_dim=1,
            x_init=state
        )

        reconstructor = StateReconstructor(dt=dt)
        checkpointer = CheckPointer(detection_delay=50)

        # Anomaly Detector
        cusum: CUSUMDetector = config['anomaly-detector']
        chi2 = ChiSquaredDetector()
        ld = LikelihoodDetector(threshold=0.995)

        under_attack = False
        prev_under_attack = False
        # Save metrics
        tracker = MetricsTracker()
        logger = Logger()
        
        prev_inputs = []
        seq_len = 50
        # Simulate
        for t in time:
            
            try:
                mission_index = mission_times.index(round(t, 1))
                target_velocity = mission[mission_index][1]
            except:
                pass
                
            # Predict state
            estimated_state = detection_ekf.get_state_estimate()
            estimated_state_variance = detection_ekf.get_state_estimate_covariance()
            
            # Take measurement
            measurement = self.measurement_func(state) + np.random.normal(measurement_noise_cov)
            
            # Launch Attack at some timesteps
            if t > attack_start and t < attack_end:
                measurement = self.attack(measurement, magnitude=config['attack-magnitude'])
            
            # Update the measurement tracking EKF
            detection_ekf.update(np.array([measurement]))

            # Calculate residual and send to anomaly detector
            residual = self.measurement_func(detection_ekf.get_state_estimate()) - \
                        self.measurement_func(detection_ekf_lagged.get_state_estimate()) # self.measurement_func(estimated_state) - ...
            _, change_detected = cusum.update(residual)
            
            # Construct tensor to feed the NN controller
            model_input, prev_inputs = self.construct_input_to_model(
                det_est_pos=detection_ekf.get_state_estimate()[0],
                det_est_vel=detection_ekf.get_state_estimate()[1],
                measured_vel=measurement,
                reference_vel=target_velocity,
                residual=residual,
                seq_len=seq_len,
                scaler=scaler,
                prev_input=prev_inputs
            )

            if change_detected:
                under_attack = not under_attack
                cusum.reset()
            # under_attack = ld.update(residual)
            # under_attack = chi2.update(residual, 1.1667472993160948)
            # Derive control input
            throttle = v_controller(model_input)
            throttle = throttle[0][0].detach().numpy() + 0
            
            if not recovery_module_on:
                under_attack = False

            if under_attack:
                
                if not prev_under_attack:
                    estimated_state, _ = reconstructor.reconstruct_state(checkpointer, ekf)
                else:
                    # Compute control based on estimated state
                    estimated_state = ekf.get_state_estimate()
                    estimated_state_variance = ekf.get_state_estimate_covariance()
                est_measurement = self.measurement_func(estimated_state)
                throttle = v_controller.compute(target_velocity, est_measurement, dt)
                # Prediction step but no correction step due to attacked sensor measurement
                ekf.predict([throttle], dt, params)
                
            else:
                # Prediction and Correction if no attack
                ekf.predict([throttle], dt, params)
                ekf.update(np.array([measurement]))
            
            # Update Checkpointer
            checkpointer.update_checkpoint(estimated_state, estimated_state_variance, [throttle])
            
            # Always Prediction and Correction for Detection EKF
            detection_ekf.predict([throttle], dt, params)
            detection_ekf_lagged.predict([throttle], dt, params)
            detection_ekf_lagged.update(np.array([measurement]))
            # detection_ekf.update(np.array([measurement]))

            # Compute next state
            derivatives = self.dynamics(state, [throttle], params) 
            tracker.track(state, estimated_state, measurement, throttle, residual, under_attack)
            # Log data
            logger.log_data({
                "time": t,
                "pos": state[0],
                "vel": state[1],
                "est_pos": estimated_state[0],
                "est_vel": estimated_state[1],
                "det_est_pos": detection_ekf.get_state_estimate()[0],
                "det_est_vel": detection_ekf.get_state_estimate()[1],
                "measured_vel": measurement,
                "reference_vel": target_velocity,
                "ctl_signal": throttle,
                "attack": attack_start < t < attack_end,
                "attack_pred": under_attack,
                "residual": residual,
                "cusum_stat": cusum.get_cusum_statistic()
            })
            state = state + derivatives * dt + np.random.multivariate_normal(np.zeros(2), process_noise_cov)
            prev_under_attack = under_attack


        # print(tracker.residual_statistics())
        # print(f"Mean Squared Control Error: {np.mean(tracker.ms_control_error())}")
        if save_data:
            logger.save_data("car")
        if show_plots:
            plot_figs(time, tracker)
            # tracker.plot_attack_predictions()
            # # Plot the road slope as a function of position
            plt.figure(figsize=(6, 4))
            positions = np.arange(0, 1000, dt)
            slopes = np.array([self.road_slope(pos) for pos in positions])
            plt.plot(positions, slopes, label="Road Slope (rad)")
            plt.xlabel("Position (m)")
            plt.ylabel("Slope (rad)")
            plt.title("Road Slope Profile")
            plt.legend()

            plt.show()

        return logger
    

    from typing import List, Tuple
    def construct_input_to_model(self, 
                                 det_est_pos: float, 
                                 det_est_vel: float, 
                                 measured_vel: float, 
                                 reference_vel: float, 
                                 residual: float, 
                                 seq_len: int,
                                 scaler: MinMaxScaler,
                                 prev_input: List[Tuple[float, float, float, float, float]]):
        
        """
        Prepares a new data point for the neural network controller


        """
        new_observation = [[det_est_pos, det_est_vel, measured_vel, reference_vel, residual]] # Not an efficient way of contructing these tensors
        new_observation_scaled = scaler.transform(new_observation)[0]
        if len(prev_input) < seq_len:
            prev_input.append(new_observation_scaled)
        elif len(prev_input) == seq_len:
            prev_input.pop(0)
            prev_input.append(new_observation_scaled)
        else:
            raise ValueError("prev_input is longer then seq_len")
        

        inputs = torch.tensor(prev_input, dtype=torch.float32).unsqueeze(0)
        return inputs, prev_input
    

if __name__ == "__main__":
    env = HillClimbingCar()

    logger = env.run_control_system_torch()


