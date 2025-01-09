import numpy as np
import matplotlib.pyplot as plt

from pid_controller import PIDController
from ekf import EKF
from anomaly_detection import CUSUMDetector
from metrics import MetricsTracker, plot_figs

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
attack_start = -1
attack_end = -1

# Controller init and target velocity
v_controller = PIDController(0.5, 0.5, 0.01)
target_velocity = 20

# Process and measurement noise
process_noise_cov = np.diag([0.01, 0.1])
measurement_noise_cov = 0.2

# Init Extended Kalman Filter
ekf = EKF(
    dynamics_func=vehicle_dynamics,
    measurement_func=measurement_func,
    state_dim=2,
    meas_dim=1,
    x_init=state
)

# Anomaly Detector
cusum = CUSUMDetector(
    thresholds=5*np.array([0.16000361755278]),
    b=np.array([0.18543593999687008]) + 0.5*np.array([0.16000361755278])
)

# Save metrics
tracker = MetricsTracker()

# Simulate
for t in time:
    estimated_state = ekf.get_state_estimate()
    residual = measurement - measurement_func(estimated_state)
    measurement = measurement_func(state) + np.random.normal(0, measurement_noise_cov)
    throttle = v_controller.compute(target_velocity, measurement, dt)
    derivatives = vehicle_dynamics(state, [throttle], params) 
    tracker.track(state, estimated_state, measurement, throttle, residual)
    state = state + derivatives * dt + np.random.multivariate_normal(np.zeros(2), process_noise_cov)
    if t > attack_start and t < attack_end:
        measurement = attack(measurement, magnitude=2.0)
        ekf.predict([throttle], dt, params)
    else:
        ekf.predict([throttle], dt, params)
        ekf.update(np.array([measurement]))
    

print(tracker.residual_statistics())
plot_figs(time, tracker)

# Plot the road slope as a function of position
plt.figure(figsize=(6, 4))
states = np.array(tracker.get_metrics()[0])
positions = states[:, 0]
slopes = np.array([road_slope(pos) for pos in positions])
plt.plot(positions, slopes, label="Road Slope (rad)")
plt.xlabel("Position (m)")
plt.ylabel("Slope (rad)")
plt.title("Road Slope Profile")
plt.legend()

plt.show()
