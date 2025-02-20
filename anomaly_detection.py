import numpy as np
from scipy.stats import chi2, foldnorm
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
    
    def reset(self):
        self.s = np.zeros_like(self.s)

    def get_cusum_statistic(self):
        if len(self.s) == 1:
            return self.s[0]
        return self.s
    

class CUSUMARDetector:
    def __init__(self, thresholds: np.ndarray, b: np.ndarray, ar_models: np.ndarray):
        self.s = np.zeros_like(thresholds)
        self.b = b
        self.ar_models = ar_models
        self.past_n_residuals = np.zeros_like(ar_models)
    
    def update(self, residuals):
        errors = residuals - (self.ar_models*self.past_n_residuals).sum(axis=1)
        


class ChiSquaredDetector:
    def __init__(self, alpha=0.01):
        self.threshold = chi2.ppf(1-alpha, df=1)

    def update(self, residual, variance):
        chi2_value = residual**2 / variance
        return chi2_value > self.threshold
    
class LikelihoodDetector:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.dist = foldnorm(c=0.1928807536992423, loc=2.421456759549301e-07, scale=1.1469198514494972)
    
    def update(self, res):
        return self.dist.cdf(np.absolute(res)) > self.threshold

class LikelihoodRatioDetector:
    def __init__(self, threshold=1.2):
        self.threshold = threshold
        dist = foldnorm(c=0.1928807536992423, loc=2.421456759549301e-07, scale=1.1469198514494972)
        
    def update(self, res):
        pass