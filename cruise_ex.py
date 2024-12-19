import numpy as np
from pid_controller import PIDController
from ekf import EKF

def road_slope(x):
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

def vehicle_dynamics(state, inputs, params):
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
    throttle = np.clip(inputs[0], 0, 1)
    
    # Extract parameters
    m = params.get("m", 1200)       # Vehicle mass (kg)
    g = params.get("g", 9.81)       # Gravity (m/s^2)
    Cr = params.get("Cr", 0.01)     # Rolling resistance coefficient
    Cd = params.get("Cd", 0.3)      # Aerodynamic drag coefficient
    A = params.get("A", 2.2)        # Frontal area (m^2)
    rho = params.get("rho", 1.225)  # Air density (kg/m^3)
    Fmax = params.get("Fmax", 4000) # Maximum engine force (N)
    
    # Compute the road slope
    theta = road_slope(x)
    
    # Forces
    F_engine = throttle * Fmax        # Engine force
    F_gravity = m * g * np.sin(theta) # Gravity force along the slope
    F_rolling = m * g * Cr            # Rolling resistance
    F_drag = 0.5 * rho * Cd * A * v**2  # Aerodynamic drag
    
    # Acceleration
    F_total = F_engine - (F_gravity + F_rolling + F_drag)
    a = F_total / m  # Newton's second law
    
    return np.array([v, a])

def measurement_func(state):
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

def attack(measurement, magnitude=0.3):
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

# Initial state and parameters
state = np.array([0, 5])  # [Position (m), Velocity (m/s)]
measurement = measurement_func(state)
params = {
    "m": 1200,       # Mass (kg)
    "g": 9.81,       # Gravity (m/s^2)
    "Cr": 0.01,      # Rolling resistance coefficient
    "Cd": 0.3,       # Aerodynamic drag coefficient
    "A": 2.2,        # Frontal area (m^2)
    "rho": 1.225,    # Air density (kg/m^3)
    "Fmax": 40000,    # Maximum engine force (N)
}

dt = 0.1        # Time step
time = np.arange(0, 50, dt)
attack_start = 10
attack_end = 20

# Controller init and target velocity
v_controller = PIDController(0.5, 0.1, 0)
target_velocity = 20

# Process and measurement noise
process_noise_cov = np.diag([0.01, 0.1])
measurement_noise_cov = 0.2

# EKF
ekf = EKF(
    dynamics_func=vehicle_dynamics,
    measurement_func=measurement_func,
    state_dim=2,
    meas_dim=1
)

# Simulate
states = []
measurements = []
throttles = []
for t in time:
    states.append(state)
    measurements.append(measurement)
    measurement = measurement_func(state) + np.random.normal(0, measurement_noise_cov)
    if t > attack_start and t < attack_end:
        measurement = attack(measurement, magnitude=2.0)
    throttle = v_controller.compute(target_velocity, measurement, dt)
    # ekf.predict([throttle], dt, params)
    # ekf.update(np.array([measurement]))
    derivatives = vehicle_dynamics(state, [throttle], params) 
    state = state + derivatives * dt + np.random.multivariate_normal(np.zeros(2), process_noise_cov)
    throttles.append(throttle)

states = np.array(states)
measurements = np.array(measurements)
throttles = np.array(throttles)
import matplotlib.pyplot as plt

# Plot position and velocity
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time, states[:, 0], label="Position (m)")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Vehicle Position")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, states[:, 1], label="True Velocity (m/s)", color='orange')
plt.plot(time, measurements, label="Measurement", color='red')
plt.vlines(10, 0, 30)
plt.vlines(20, 0, 30)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Vehicle Velocity")
plt.legend()

plt.figure(figsize=(6, 4))
plt.plot(time, throttles, label="Throttle")
plt.xlabel("Time (s)")
plt.ylabel("Throttle")
plt.legend()

# Plot the road slope as a function of position
plt.figure(figsize=(6, 4))
positions = states[:, 0]
slopes = np.array([road_slope(pos) for pos in positions])
plt.plot(positions, slopes, label="Road Slope (rad)")
plt.xlabel("Position (m)")
plt.ylabel("Slope (rad)")
plt.title("Road Slope Profile")
plt.legend()

plt.show()
