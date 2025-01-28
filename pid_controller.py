import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, integral_windup=20000.0, dead_zone=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

        self.integral_windup_threshold = integral_windup
        self.dead_zone = dead_zone
    
    def compute(self, target, current, dt):
        error = target - current
        if np.absolute(self.integral + error*dt) < self.integral_windup_threshold:
            self.integral += error * dt
        else:
            self.integral = 0.0
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(u, -1, 1)
    
    def get_integral(self):
        return self.integral