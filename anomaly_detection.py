import numpy as np
from scipy.stats import chi2
from collections import deque

class CUSUMDetector:
    def __init__(self, thresholds, b, epsilon=0.1):
        self.s = np.zeros_like(thresholds)
        self.thresholds = thresholds
        self.b = b
        self.eps = epsilon
        

    def update(self, residuals):
        """
        Updates the CUSUM statistic and returns indices of sensors over
        the thresholds.

        Parameters:
        -----------
            residuals : np.array
                        residuals from relevant sensors
        Updates:
        --------
            self.s : np.array
                     CUSUM statistic
        Returns:
        --------
            np.array : sensor indices for which the CUSUM statistic has
                       exceeded the threshold
            bool : True if any of the sensors are above its threshold
        """
        self.s = np.maximum(self.s + np.absolute(residuals) - self.b, np.zeros_like(self.s))

        alarm_indices = np.where(self.s > self.thresholds)
        self.s[alarm_indices] = self.thresholds[alarm_indices] + self.eps*np.ones_like(self.s[alarm_indices])
        return np.where(self.s > self.thresholds)[0], np.any(self.s > self.thresholds)
    
class ChiSquaredDetector:
    def __init__(self, alpha=0.05):
        self.threshold = chi2.ppf(1-alpha, df=1)

    def update(self, residual, variance):
        chi2_value = residual**2 / variance
        return chi2_value > self.threshold