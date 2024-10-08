import numpy as np
from abc import ABC, abstractmethod

class AbstractConfigurator(ABC):
    """
    Abstract class for dynamic online configuration.
    """

    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.

        :param state: The current state.
        :return: The selected action.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Learn from the experience tuple.

        :param state: The current state.
        :param action: The selected action.
        :param reward: The received reward.
        :param next_state: The next state.
        :param done: Whether the episode is done.
        """
        raise NotImplementedError