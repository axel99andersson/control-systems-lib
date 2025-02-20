import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

class MetricsTracker:
    def __init__(self):
        self.states = []
        self.measurements = []
        self.state_estimates = []
        self.control_signals = []
        self.residuals = []
        self.attack_predictions = []
        self.metrics_map = {
            "states": self.states,
            "measurements": self.measurements,
            "state_estimates": self.state_estimates,
            "control_signals": self.control_signals,
            "residuals": self.residuals,
            "attack_predictions": self.attack_predictions,
        }

    def track(self, state, estimated_state, measurement, control_signal, residuals, attack_prediction=None):
        self.states.append(state)
        self.state_estimates.append(estimated_state)
        self.measurements.append(measurement)
        self.control_signals.append(control_signal)
        self.residuals.append(residuals)
        self.attack_predictions.append(attack_prediction)
        self.metrics_map = {
            "states": self.states,
            "measurements": self.measurements,
            "state_estimates": self.state_estimates,
            "control_signals": self.control_signals,
            "residuals": self.residuals,
            "attack_predictions": self.attack_predictions
        }

    def get_metrics(self, metric=None):
        if metric is None:
            return [self.states, 
                    self.state_estimates, 
                    self.measurements, 
                    self.control_signals, 
                    self.residuals,
                    self.attack_predictions]
        else:
            if metric not in self.metrics_map.keys(): raise Exception(f"{metric} not a Metric")
            return self.metrics_map[metric]
    
    def residual_statistics(self):
        abs_residuals = np.absolute(self.residuals)
        mean, std = abs_residuals.mean(), abs_residuals.std()
        return mean, std
    
    def ms_control_error(self):
        states = np.array(self.states)
        velocities = states[:,1]
        setpoints = 20*np.ones_like(velocities)
        squared_errors = (velocities - setpoints)**2
        return squared_errors
    
    def plot_attack_predictions(self):
        plt.figure(figsize=(6,4))
        plt.plot(np.array(range(len(self.attack_predictions))) / 10, self.attack_predictions)
        plt.xlabel("Time step")
        plt.ylabel("Attack Prediction")

class Logger:
    def __init__(self):
        """
        Stores data for each time step, each time step is a dictionary
        """
        self.data = []

    def log_data(self, data: dict):  
        self.data.append(data)

    def save_data(self, control_system: str, filename: str = None):
        if filename is None:
            filename = f"{control_system}_data_{dt.datetime.now()}".replace(" ", "")
        
        df = pd.DataFrame(self.data)
        df.to_csv(f"./data/{filename}.csv")


def plot_figs(time, tracker: MetricsTracker):
    states, state_estimates, measurements, control_signals, residuals, attack_predictions = tracker.get_metrics()

    states = np.array(states)
    state_estimates = np.array(state_estimates)
    measurements = np.array(measurements)
    throttles = np.array(control_signals)

    # Plot position and velocity
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(time, states[:, 0], label="Position (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Vehicle Position")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, states[:, 1], label="True Velocity (m/s)", color='orange')
    # plt.plot(time, np.concatenate((20*np.ones(2000), 21*np.ones(8000))), label="Reference", color="blue")
    plt.plot(time, 20*np.ones_like(time), label="Reference", color="blue")
    # plt.plot(time, measurements, label="Measurement", color='red')
    # plt.plot(time, state_estimates[:,1], label="Estimated Velocity", color='blue')
    plt.vlines(10, 0, 30)
    plt.vlines(20, 0, 30)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Vehicle Velocity")
    plt.legend()

    # plt.figure(figsize=(6, 4))
    # plt.plot(time, throttles, label="Throttle")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Throttle")
    # plt.legend()

    # plt.figure(figsize=(6, 4))
    # plt.plot(time, np.absolute(residuals), label="Absolute Residuals")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Residual")
    # plt.legend()
