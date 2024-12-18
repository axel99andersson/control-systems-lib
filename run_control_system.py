import numpy as np
import matplotlib.pyplot as plt
import control as ct
from math import pi

from update_functions import vehicle_update, attacker_output

# Define the input/output system for the vehicle
vehicle = ct.NonlinearIOSystem(
    vehicle_update, None, name='vehicle',
    inputs=('u', 'gear', 'theta'), outputs=('v'), states=('v'))

def pi_update(t, x, u, params={}):
    # Get the controller parameters that we need
    ki = params.get('ki', 0.1)
    kaw = params.get('kaw', 2)  # anti-windup gain

    # Assign variables for inputs and states (for readability)
    v = u[0]                    # current velocity
    vref = u[1]                 # reference velocity
    z = x[0]                    # integrated error

    # Compute the nominal controller output (needed for anti-windup)
    u_a = pi_output(t, x, u, params)

    # Compute anti-windup compensation (scale by ki to account for structure)
    u_aw = kaw/ki * (np.clip(u_a, 0, 1) - u_a) if ki != 0 else 0

    # State is the integrated error, minus anti-windup compensation
    return (vref - v) + u_aw

def pi_output(t, x, u, params={}):
    # Get the controller parameters that we need
    kp = params.get('kp', 0.5)
    ki = params.get('ki', 0.1)

    # Assign variables for inputs and states (for readability)
    v = u[0]                    # current velocity
    vref = u[1]                 # reference velocity
    z = x[0]                    # integrated error

    # PI controller
    return kp * (vref - v) + ki * z

control_pi = ct.NonlinearIOSystem(
    pi_update, pi_output, name='control',
    inputs=['v', 'vref'], outputs=['u'], states=['z'],
    params={'kp': 0.5, 'ki': 0.1})

def attacker_update(t, x, u, params={}):
    """Attacker does not have states."""
    return []  # No state variables

def attacker_output(t, x, u, params={}):
    """Output the attacked measurement signal."""
    v = u[0]                     # Actual velocity
    attack_strength = params.get('attack_strength', 0.1)
    return v + attack_strength # * np.sin(2 * np.pi * t)  # Add a sinusoidal disturbance

attacker = ct.NonlinearIOSystem(
    attacker_update, attacker_output, name="attacker",
    inputs=["v"], outputs=["v_attacked"], states=[], params={"attack_strength": 0.1}
)

def delay_update(t, x, u, params={}):
    """Update function for the delay system."""
    return u[0]  # Store the current input as the state for the next step

def delay_output(t, x, u, params={}):
    """Output function for the delay system."""
    return x[0]  # Output the delayed state

delay_system = ct.NonlinearIOSystem(
    delay_update, delay_output, name="delay",
    inputs=["v"], outputs=["v_delayed"], states=["x"]
)

interconnected_attacked_system = ct.InterconnectedSystem(
    (vehicle, control_pi, attacker, delay_system),
    connections=(
        # Control system connections
        ("control.v", "attacker.v_attacked"),   # Control receives the attacked velocity
        ("vehicle.u", "control.u"),            # Vehicle receives the control signal

        # Attacker and delay connections
        ("attacker.v", "delay.v_delayed"),     # Attacker receives delayed velocity
        ("delay.v", "vehicle.v"),             # Delay input is the vehicle velocity

    ),
    inplist=["control.vref", "vehicle.gear", "vehicle.theta"],
    outlist=["vehicle.v", "attacker.v_attacked", "control.u"],
    outputs=['v', 'va', 'u']
)

cruise_pi = ct.InterconnectedSystem(
    (vehicle, control_pi), name='cruise',
    connections=[
        ['vehicle.u', 'control.u'],
        ['control.v', 'vehicle.v']],
    inplist=['control.vref', 'vehicle.gear', 'vehicle.theta'],
    outlist=['control.u', 'vehicle.v'], outputs=['u', 'v'])

# Plot function
def cruise_plot(sys, t, y, label=None, t_hill=None, vref=20, antiwindup=False,
                linetype='b-', subplots=None, legend=None):
    if subplots is None:
        subplots = [None, None]
    # Figure out the plot bounds and indices
    v_min = vref-1.2; v_max = vref+0.5; v_ind = sys.find_output('v')
    u_min = 0; u_max = 2 if antiwindup else 1; u_ind = sys.find_output('u')

    # Make sure the upper and lower bounds on v are OK
    while max(y[v_ind]) > v_max: v_max += 1
    while min(y[v_ind]) < v_min: v_min -= 1

    # Create arrays for return values
    subplot_axes = list(subplots)

    # Velocity profile
    if subplot_axes[0] is None:
        subplot_axes[0] = plt.subplot(2, 1, 1)
    else:
        plt.sca(subplots[0])
    plt.plot(t, y[v_ind], linetype)
    plt.plot(t, vref*np.ones(t.shape), 'k-')
    if t_hill:
        plt.axvline(t_hill, color='k', linestyle='--', label='t hill')
    plt.axis([0, t[-1], v_min, v_max])
    plt.xlabel('Time $t$ [s]')
    plt.ylabel('Velocity $v$ [m/s]')

    # Commanded input profile
    if subplot_axes[1] is None:
        subplot_axes[1] = plt.subplot(2, 1, 2)
    else:
        plt.sca(subplots[1])
    plt.plot(t, y[u_ind], 'r--' if antiwindup else linetype, label=label)
    # Applied input profile
    if antiwindup:
        # TODO: plot the actual signal from the process?
        plt.plot(t, np.clip(y[u_ind], 0, 1), linetype, label='Applied')
    if t_hill:
        plt.axvline(t_hill, color='k', linestyle='--')
    if legend:
        plt.legend(frameon=False)
    plt.axis([0, t[-1], u_min, u_max])
    plt.xlabel('Time $t$ [s]')
    plt.ylabel('Throttle $u$')

    return subplot_axes

# Simulate the system
time = np.linspace(0, 30, 101)
vref = 20 * np.ones_like(time)  # Reference velocity (25 m/s)
gear = 4 * np.ones_like(time)   # Constant gear (3rd gear)
theta = np.zeros_like(time)     # Flat road (theta = 0)
theta_hill = [
    0 if t <= 5 else
    4./180. * pi * (t-5) if t <= 6 else
    4./180. * pi for t in time]
inputs = [vref, gear, theta_hill]
X0, U0, Y0 = ct.find_eqpt(
    cruise_pi, [vref[0], 0], [vref[0], gear[0], theta[0]],
    y0=[0, vref[0]], iu=[1, 2], iy=[1], return_y=True)
breakpoint()
t, y = ct.input_output_response(interconnected_attacked_system, time, inputs, 20)
plt.figure()
plt.suptitle('Car with cruise control encountering sloping road')
cruise_plot(interconnected_attacked_system, t, y, t_hill=5)
plt.show()