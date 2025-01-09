import numpy as np
from pid_controller import PIDController

def quadcopter_dynamics(state, inputs, params):
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

# Initial conditions and parameters
state = np.zeros(12)  # Initial state [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
params = {"m": 1.5, "g": 9.81, "l": 0.25, "Ixx": 0.0142, "Iyy": 0.0142, "Izz": 0.0284, "k": 0.01}
dt = 0.01  # Time step
time = np.arange(0, 30, dt)  # Simulation time

# PID controllers
altitude_pid = PIDController(0.15, 0.05, 0.12)
roll_pid = PIDController(1.0, 0.1, 0.01)
pitch_pid = PIDController(1.0, 0.1, 0.01)
yaw_pid = PIDController(1.0, 0.1, 0.01)

# Target states
target_altitude = 10  # Target altitude (m)
target_roll = 0       # Target roll (rad)
target_pitch = 0      # Target pitch (rad)
target_yaw = 0        # Target yaw (rad)

# Simulation
states = []
for t in time:
    # Extract current state
    z, phi, theta, psi = state[2], state[6], state[7], state[8]
    
    # Compute PID outputs
    Fz = altitude_pid.compute(target_altitude, z, dt)
    tau_phi = roll_pid.compute(target_roll, phi, dt)
    tau_theta = pitch_pid.compute(target_pitch, theta, dt)
    tau_psi = yaw_pid.compute(target_yaw, psi, dt)
    # Convert to rotor forces
    f1 = (Fz + tau_theta - tau_phi + tau_psi) / 4
    f2 = (Fz - tau_theta - tau_phi - tau_psi) / 4
    f3 = (Fz + tau_theta + tau_phi - tau_psi) / 4
    f4 = (Fz - tau_theta + tau_phi + tau_psi) / 4
    inputs = np.clip([f1, f2, f3, f4], 0, np.inf)  # Ensure forces are non-negative
    
    # Update state
    state = state + quadcopter_dynamics(state, inputs, params) * dt
    state[2] = max(state[2], 0) # Altitude correction
    states.append(state)

import matplotlib.pyplot as plt

states = np.array(states)
plt.figure()
plt.plot(time, states[:, 2], label='Altitude (z)')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.show()

