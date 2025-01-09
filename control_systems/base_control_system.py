from abc import ABC, abstractmethod

class BaseControlSystem(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def dynamics(self, state, inputs, params):
        pass

    @abstractmethod
    def measurement_func(self, state):
        pass

    @abstractmethod
    def attack(self, measurement):
        pass

    @abstractmethod
    def run_control_system(self, config):
        pass