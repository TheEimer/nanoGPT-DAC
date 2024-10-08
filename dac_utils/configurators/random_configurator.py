import numpy as np
from dac_utils.configurators.abstract_configurator import AbstractConfigurator

class RandomConfigurator(AbstractConfigurator):
    """
    Random sampling agent.
    """

    def __init__(self, upper, lower, dist_name=None, dist_kwargs=None):
        if dist_name is None:
            dist_name = "uniform"
        
        self.rng = getattr(np.random.default_rng(), dist_name)
        self.lower = lower
        self.upper = upper
        self.dist_kwargs = dist_kwargs

    def predict(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.

        :param state: The current state.
        :return: The selected action.
        """
        return self.rng(self.lower, self.upper, **self.dist_kwargs)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Learn from the experience tuple.

        :param state: The current state.
        :param action: The selected action.
        :param reward: The received reward.
        :param next_state: The next state.
        :param done: Whether the episode is done.
        """
        pass