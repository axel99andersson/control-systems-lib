import numpy as np

from pid_controller import PIDController
from .base_control_system import BaseControlSystem
from anomaly_detection import CUSUMDetector
from ekf import EKF
from metrics import MetricsTracker

class QuadcopterAltitude(BaseControlSystem):

    def __init__(self):
        super().__init__()
    
    def dynamics(self, state, inputs, params):
        """
        Compute the state derivatives for a quadcopter.
        
        Parameters:
            state: np.array [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
                - Position: [x, y, z]
                - Velocity: [vx, vy, vz]
                - Orientation: [phi (roll), theta (pitch), psi (yaw)]
                - Angular velocity: [p, q, r]
            inputs: np.array [f1, f2, f3, f4]
                - Rotor forces (f1 to f4)
            params: dict containing physical parameters of the quadcopter.
        """
        
        # Extract parameters
        m = params.get("m", 1.5)    # Mass (kg)
        g = params.get("g", 9.81)   # Gravity (m/s^2)
        l = params.get("l", 0.25)   # Arm length (m)
        Ixx = params.get("Ixx", 0.0142)  # Moment of inertia about x-axis
        Iyy = params.get("Iyy", 0.0142)  # Moment of inertia about y-axis
        Izz = params.get("Izz", 0.0284)  # Moment of inertia about z-axis
        k = params.get("k", 0.01)  # Torque coefficient
        b = params.get("b", 75) # Max thrust force (N)
        
        # Extract state variables
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        f1, f2, f3, f4 = inputs  # Rotor speeds

        # Compute forces and torques
        Fz = b * (f1 + f2 + f3 + f4)  # Total thrust
        tau_phi = b*l * (f1 - f3)  # Roll torque
        tau_theta = b*l * (f2 - f4)  # Pitch torque
        tau_psi = k * (f1 - f2 + f3 - f4)  # Yaw torque
        
        # Rotation matrix
        R = np.array([
            [np.cos(phi)*np.cos(psi) - np.cos(theta)*np.sin(phi)*np.sin(psi), 
            -np.cos(psi)*np.sin(phi) - np.cos(phi)*np.cos(theta)*np.sin(psi),
            np.sin(theta)*np.sin(psi)],
            [np.cos(theta) * np.cos(phi)*np.sin(psi) + np.cos(phi)*np.sin(psi), 
            np.cos(phi)*np.cos(theta)*np.cos(psi) - np.sin(phi)*np.sin(psi),
            -np.cos(psi)*np.sin(theta)],
            [np.sin(phi)*np.sin(theta),
             np.cos(phi) * np.sin(theta), 
             np.cos(theta)]
        ])
        
        # Linear accelerations
        acc = (R @ np.array([0, 0, Fz])) / m - np.array([0, 0, g]) if z >= 0 else (R @ np.array([0, 0, Fz]))
        
        # Angular accelerations
        p_dot = (tau_phi - (Iyy - Izz) * q * r) / Ixx
        q_dot = (tau_theta - (Izz - Ixx) * p * r) / Iyy
        r_dot = (tau_psi - (Ixx - Iyy) * p * q) / Izz
        
        # State derivatives
        return np.array([
            vx, vy, vz,                 # Position derivatives
            acc[0], acc[1], acc[2],     # Velocity derivatives
            p, q, r,                   # Orientation derivatives
            p_dot, q_dot, r_dot         # Angular velocity derivatives
        ])
    
    def measurement_func(self, state):
        """
        Measurement function, mapping states to measurements
        The altitude and Euler angles are measured in this setup

        Parameters:
            state: np.array
    
        Returns:
            np.array: [z (alt.), phi, theta, psi (Euler ang.)]
        """
        z, phi, theta, psi = state[2], state[6], state[7], state[8]
        return np.array([z, phi, theta, psi])
    
    def attack(self, measurement, magnitude):
        """
        Attack the measurement signal with some magnitude
        TODO: Implement more types of attacks

        Parameters:
            measurement: float
            magnitude: float
        Returns:
            float
        """
        attacked_measurement = measurement + magnitude*np.array([1, 0, 0, 0]) # Attack on altitude
        return attacked_measurement

    def run_control_system(self, config=
        {
            "params": {"m": 1.5, 
                        "g": 9.81, 
                        "l": 0.25, 
                        "Ixx": 0.0142, 
                        "Iyy": 0.0142, 
                        "Izz": 0.0284, 
                        "k": 0.01,
                        "b": 75
                    },
            "init-state": np.zeros(12),
            "dt": 0.01,
            "time": 30,
            "attack-start": -1,
            "attack-end": -1,
            "altitude-pid": PIDController(0.15, 0.05, 0.12),
            "roll-pid": PIDController(0.15, 0.05, 0.12),
            "pitch-pid": PIDController(0.15, 0.05, 0.12),
            "yaw-pid": PIDController(0.15, 0.05, 0.12),
            "target-altitude": 10,
            "target-roll": 0,
            "target-pitch": 0,
            "target-yaw": 0,
            "process-noise-cov": np.diag([0.01 for _ in range(12)]),
            "measurement-noise-cov": np.diag([0.000000005 for _ in range(4)]),
            "anomaly-detector": CUSUMDetector(
                thresholds=5*np.array([0.16000361755278]),
                b=np.array([0.18543593999687008]) + 0.5*np.array([0.16000361755278])),
        }):

        # Process and measurement noise
        process_noise_cov = config['process-noise-cov']
        measurement_noise_cov = config['measurement-noise-cov']

        # Initial conditions and parameters
        state = config['init-state']  # Initial state [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        # measurement = self.measurement_func(state) + np.random.multivariate_normal(np.zeros(4), measurement_noise_cov)
        params = config['params']
        
        dt = config['dt']  # Time step
        time = np.arange(0, config['time'], dt)  # Simulation time
        attack_start = config['attack-start']
        attack_end = config['attack-end']

        # PID controllers
        altitude_pid: PIDController = config['altitude-pid']
        roll_pid: PIDController = config['roll-pid']
        pitch_pid: PIDController = config['pitch-pid']
        yaw_pid: PIDController = config['yaw-pid']

        # Target states
        target_altitude = config['target-altitude']  # Target altitude (m)
        target_roll = config['target-roll']       # Target roll (rad)
        target_pitch = config['target-pitch']      # Target pitch (rad)
        target_yaw = config['target-yaw']        # Target yaw (rad)

        # Init Extended Kalman Filter
        ekf = EKF(
            dynamics_func=self.dynamics,
            measurement_func=self.measurement_func,
            state_dim=12,
            meas_dim=4,
            x_init=state
        )

        # Anomaly Detector
        cusum = config['anomaly-detector']

        # Save metrics
        tracker = MetricsTracker()

        # Simulation
        for t in time:
            estimated_state = ekf.get_state_estimate()
            residual = 1# measurement - self.measurement_func(estimated_state)
            # Extract current state
            
            z, phi, theta, psi = self.measurement_func(state) \
                + np.random.multivariate_normal(np.zeros(4), measurement_noise_cov)
            z = max(0, z)
            measurement = np.array([z, phi, theta, psi])
            
            # Compute PID outputs
            Fz = altitude_pid.compute(target_altitude, z, dt)
            tau_phi = roll_pid.compute(target_roll, phi, dt)
            tau_theta = pitch_pid.compute(target_pitch, theta, dt)
            tau_psi = yaw_pid.compute(target_yaw, psi, dt)
            # Convert to rotor forces
            # f1 = (Fz + tau_theta - tau_phi + tau_psi) / 4
            # f2 = (Fz - tau_theta - tau_phi - tau_psi) / 4
            # f3 = (Fz + tau_theta + tau_phi - tau_psi) / 4
            # f4 = (Fz - tau_theta + tau_phi + tau_psi) / 4
            torques = np.array([Fz, tau_phi, tau_theta, tau_psi])
            b, l, k = params.get('b'), params.get('l'), params.get('k')
            rotorspeed_to_torque = np.array([
                [b, b, b, b],
                [b*l, 0, -b*l, 0],
                [0, b*l, 0, -b*l],
                [k, -k, k, -k]
            ])
            breakpoint()
            inputs = np.linalg.solve(rotorspeed_to_torque, torques)
            inputs = np.clip(inputs, 0, np.inf)  # Ensure forces are non-negative
            # Log
            tracker.track(state, estimated_state, measurement, inputs, residual)
            # Update state
            state = state + self.dynamics(state, inputs, params) * dt \
                # + np.random.multivariate_normal(np.zeros_like(state), process_noise_cov)
            state[2] = max(state[2], 0) # Altitude correction
            
            if t > attack_start and t < attack_end:
                measurement = self.attack(measurement, magnitude=2.0)
                ekf.predict(inputs, dt, params)
            else:
                ekf.predict(inputs, dt, params)
                ekf.update(measurement)

        import matplotlib.pyplot as plt

        states = np.array(tracker.get_metrics(metric='states'))
        plt.figure()
        plt.plot(time, states[:, 2], label='Altitude (z)')
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.legend()
        plt.show()