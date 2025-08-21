import numpy as np
from pid_controller import PIDController
from ekf import EKF
from anomaly_detection import CUSUMDetector

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
state[2] = 10.0
params = {"m": 1.5, "g": 9.81, "l": 0.25, "Ixx": 0.0142, "Iyy": 0.0142, "Izz": 0.0284, "k": 0.01}
dt = 0.01  # Time step
time = np.arange(0, 30, dt)  # Simulation time

# PID controllers
altitude_pid = PIDController(0.15, 0.05, 0.12)
roll_pid = PIDController(1.0, 0.1, 0.01)
pitch_pid = PIDController(1.0, 0.1, 0.01)
yaw_pid = PIDController(1.0, 0.1, 0.01)

# EKF
process_var = 0.0001
measurement_var = 0.0001
Q = process_var*np.eye(12)
R = measurement_var*np.eye(12)
ekf = EKF(dynamics_func=quadcopter_dynamics, measurement_func=lambda x: x, state_dim=12, meas_dim=12, Q=Q, R=R, x_init=state)
prior_estimate = ekf.get_state_estimate()

# CUSUM Detector
thresholds = np.array([ 0.0058917, 0.0147852, 0.003, 0.00100202, 0.00131403, 0.00012968,
                        0.00013316, 0.0001297, 0.00012684, 0.00013067, 0.0001296, 0.00013194]) + \
            50*np.array([7.13016091e-03, 1.29131511e-02, 2.16162300e-04, 8.43381443e-04, 6.60183873e-04, 9.59482238e-05, 
                        9.72591667e-05, 9.66869461e-05, 9.65746976e-05, 1.00110807e-04, 9.70683275e-05, 1.01821117e-04])

b = np.array([ 0.0058917, 0.0147852, 0.003, 0.00100202, 0.00131403, 0.00012968,
                        0.00013316, 0.0001297, 0.00012684, 0.00013067, 0.0001296, 0.00013194]) + \
            0*np.array([7.13016091e-03, 1.29131511e-02, 2.16162300e-04, 8.43381443e-04, 6.60183873e-04, 9.59482238e-05, 
                        9.72591667e-05, 9.66869461e-05, 9.65746976e-05, 1.00110807e-04, 9.70683275e-05, 1.01821117e-04])

cusum = CUSUMDetector(thresholds=thresholds, b=b)

# Target states
target_altitude = 10  # Target altitude (m)
target_roll = 0       # Target roll (rad)
target_pitch = 0      # Target pitch (rad)
target_yaw = 0        # Target yaw (rad)

f_eq = 0.04907

# Attacking
attack_start_time = 10.0
attack_magnitude = 2.0
attack_vector = np.zeros(12)
attack_vector[2] = attack_magnitude
# Logging
states = []
state_estimate_priors = []
state_estimate_posteriors = []
cusum_statistics = []

for t in time:
    # Extract current state and measurement
    z, phi, theta, psi = state[2], state[6], state[7], state[8]
    measurement = state + np.random.normal(loc=0.0, scale=measurement_var)

    # if t >= attack_start_time:
    #     measurement = measurement + attack_vector
    
    _, _, cusum_stat = cusum.update(measurement - prior_estimate)
    cusum_statistics.append(cusum_stat)
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
    inputs = np.clip([f_eq, f_eq, f_eq, f_eq], 0, np.inf)  # Ensure forces are non-negative

    # Update EKF
    ekf.predict(u=inputs, dt=dt, params=params)
    prior_estimate = ekf.get_state_estimate()
    state_estimate_priors.append(prior_estimate)
    
    ekf.update(measurement)
    state_estimate_posteriors.append(ekf.get_state_estimate())

    # Update state
    state = state + quadcopter_dynamics(state, inputs, params) * dt + np.random.normal(loc=0.0, scale=process_var, size=12)
    state[2] = max(state[2], 0) # Altitude correction
    states.append(state)

import matplotlib.pyplot as plt

states = np.array(states)
state_estimate_priors = np.array(state_estimate_priors)
state_estimate_posteriors = np.array(state_estimate_posteriors)
cusum_statistics = np.array(cusum_statistics)
plt.figure()
plt.plot(time, states[:, 2], color="darkblue", linestyle="--", label='Altitude (z)')
plt.plot(time, state_estimate_priors[:, 2], color="darkorange", linestyle="-.", label="Altitude Estimate")
plt.ylim(bottom=0.0, top=15.0)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.show()

# Plot estimation error
estimation_error_z = np.absolute(states[:,2] - state_estimate_priors[:,2])
plt.figure()
plt.plot(time, estimation_error_z, label="Estimation error (z) $(x_k-\hat{x}_{k:k-1})^2$")
plt.xlabel('Time (s)')
plt.ylabel('Estimation SE')
plt.xlim(left=9.9, right=10.2)
plt.legend()
plt.show()

# Plot CUSUM Statistic
plt.figure()
plt.plot(time, cusum_statistics[:,2], label="CUSUM Statistic")
plt.xlabel('Time (s)')
plt.ylabel('CUSUM Statistic')
plt.hlines(y=thresholds[2], xmin=np.min(time), xmax=np.max(time), color="darkorange", linestyles="--", label="Threshold")
plt.legend()
plt.show()

print("Mean Absolute Estimation Errors", np.mean(np.absolute(states - state_estimate_priors), axis=0))
print("Variance of Absolute Estimation Error", np.std(np.absolute(states - state_estimate_priors), axis=0))
