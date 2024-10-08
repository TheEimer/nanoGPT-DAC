import numpy as np
from dac_utils.configurators.abstract_configurator import AbstractConfigurator

class StaticConfigurator(AbstractConfigurator):
    """
    Static agent.
    """

    def __init__(self, action):
        self.action = action

    def predict(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.

        :param state: The current state.
        :return: The selected action.
        """
        return self.action

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