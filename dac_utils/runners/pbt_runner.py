import numpy as np
from typing import Dict
from copy import deepcopy
from abstract_runner import AbstractRunner

class PBTRunner(AbstractRunner):
    """
    Abstract class for DAC agents.
    """

    def __init__(self, submitit_args, configspace, population_size, quantiles, mutations, categorical_sample_prob):
        self.population_size = population_size
        self.quantiles = quantiles
        self.mutations = mutations
        self.categorical_sample_prob = categorical_sample_prob
        self.submitit_args = submitit_args
        self.iteration = 0

        # Separate into categoricals and non-categoricals
        self.categoricals = [c for c in configspace.items() if c.__class__.__name__ == "CategoricalHyperparameter"]
        self.non_categoricals = [c for c in configspace.items() if c.__class__.__name__ != "CategoricalHyperparameter"]

        # First run random configs
        self.next_configs = [dict(c) for c in configspace.sample_configuration(size=population_size)]
        for i, c in enumerate(self.next_configs):
            c["save_at"] = f"checkpoint_generation_0_config_{i}.pt"
            c["load_from"] = False

    def get_submitit_config(self, state: np.ndarray) -> Dict:
        """
        Get the slurm configuration for the next jobs.

        :param state: The current state.
        :return: The selected action.
        """
        return self.submitit_args

    def get_step_args(self, configurator, statistics) -> Dict:
        """
        Get arguments for the step executions.

        :param configurator: The chosen configurator during the execution.
        :param statistics: The current statistics.
        """
        return self.next_configs
    
    def set_results(self, results) -> None:
        """
        Internal update with the results from the submitted jobs.

        :param results: The results from the submitted jobs.
        """
        self.iteration += 1
        self.old_configs = deepcopy(self.next_configs)

        # Update save locations
        self.next_configs = [{"save_at": f"checkpoint_generation_{self.iteration}_config_{i}.pt"} for i in range(len(self.old_configs))]

        # Get ids of best and worst quantiles
        losses = [r["loss"] for r in results]
        upper_performance_quantile = np.quantile(losses, self.quantiles[1])
        lower_performance_quantile = np.quantile(losses, self.quantiles[0])
        worst_ids = [i for i in range(len(losses)) if losses[i] > upper_performance_quantile]
        best_ids = [i for i in range(len(losses)) if losses[i] < lower_performance_quantile]

        # Overwrite load location of worst quantiles with best quantiles
        for i in range(len(self.old_configs)):
            if i in worst_ids:
                self.next_configs[i]["load_from"] = self.old_configs[np.random.choice(best_ids)]["save_at"]
            else:
                self.next_configs[i]["load_from"] = self.old_configs[i]["save_at"]

        # Mutate ints and floats
        for i, c in enumerate(self.old_configs):
            for k in self.non_categoricals:
                if np.random.rand() < 0.5:
                    hp_value = c[k.name] * self.mutations[0]
                else:
                    hp_value = c[k.name] * self.mutations[1]
                hp_value = np.clip(hp_value, k.lower, k.upper)
                if k.__class__.__name__ == "IntegerHyperparameter":
                    hp_value = int(hp_value)
                self.next_configs[i][k.name] = hp_value

        # Resample categoricals
        for i, c in enumerate(self.old_configs):
            for k in self.categoricals:
                if np.random.rand() < self.categorical_sample_prob:
                    self.next_configs[i][k.name] = k.sample_uniform()
                else:
                    self.next_configs[i][k.name] = c[k.name]

    def save(self, path: str) -> None:
        """
        Save the agent's model.

        :param path: The path to save the model.
        """
        # TODO: load history to continue run?
        pass

    def load(self, path: str) -> None:
        """
        Load the agent's model.

        :param path: The path to load the model.
        """
        # TODO: save history to continue run?
        pass