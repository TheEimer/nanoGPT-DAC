import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class AbstractRunner(ABC):
    """
    Abstract class for DAC agents.
    """

    @abstractmethod
    def get_submitit_config(self, state: np.ndarray) -> Dict:
        """
        Get the slurm configuration for the next jobs.

        :param state: The current state.
        :return: The selected action.
        """
        raise NotImplementedError

    @abstractmethod
    def get_step_args(self, configurator, statistics) -> Dict:
        """
        Get arguments for the step executions.

        :param configurator: The chosen configurator during the execution.
        :param statistics: The current statistics.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_results(results: Dict) -> None:
        """
        Internal update with the results from the submitted jobs.

        :param results: The results from the submitted jobs.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent's model.

        :param path: The path to save the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the agent's model.

        :param path: The path to load the model.
        """
        raise NotImplementedError