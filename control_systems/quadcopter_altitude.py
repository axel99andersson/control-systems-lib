import numpy as np

from pid_controller import PIDController
from .base_control_system import BaseControlSystem

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
        Fmax = params.get("Fmax", 75) # Max thrust force (N)
        
        # Extract state variables
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        f1, f2, f3, f4 = Fmax*inputs  # Rotor forces

        # Compute forces and torques
        Fz = f1 + f2 + f3 + f4  # Total thrust
        tau_phi = l * (f2 - f4)  # Roll torque
        tau_theta = l * (f3 - f1)  # Pitch torque
        tau_psi = k * (f1 - f2 + f3 - f4)  # Yaw torque
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta) * np.cos(psi), 
            np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
            np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
            [np.cos(theta) * np.sin(psi), 
            np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
            np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
            [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]
        ])
        
        # Linear accelerations
        acc = (R @ np.array([0, 0, Fz])) / m - np.array([0, 0, g]) if z != 0 else (R @ np.array([0, 0, Fz]))
        
        # Angular accelerations
        p_dot = (tau_phi - (Izz - Iyy) * q * r) / Ixx
        q_dot = (tau_theta - (Ixx - Izz) * p * r) / Iyy
        r_dot = (tau_psi - (Iyy - Ixx) * p * q) / Izz
        
        # State derivatives
        return np.array([
            vx, vy, vz,                 # Position derivatives
            acc[0], acc[1], acc[2],     # Velocity derivatives
            p, q, r,                   # Orientation derivatives
            p_dot, q_dot, r_dot         # Angular velocity derivatives
        ])
    
    def measurement_func(self, state):
        return super().measurement_func(state)
    
    def attack(self, measurement):
        return super().attack(measurement)
    
    def run_control_system(self, config):
        return super().run_control_system(config)