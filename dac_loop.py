# 0. define what an agent looks like. Should include options for:
#    - bandit
#    - model-free RL
#    - model-based RL
#    - reptile
#    - BO Bandit (e.g. PB2)
# 1. Set up agent
# 2. Define cosine length
# 3. For total num updates:
#     4. Use hydra to launch n parallel step jobs
#        Each of them gets an agent copy
#     5. Collect performances and transitions
#     6. Update agent

# Questions: 
# - model-free (PPO) or model-based (???)
# - local agent updates or not (reptile?)
# - what is the state?
import numpy as np
from do_step import do_step
from functools import partial

from dac_utils.branching_criteria import fully_sequential
from dac_utils.termination_criteria import fixed_iterations

def train_gpt(cfg):

    # TODO: load these as functions instead of using the string
    # Get the branching and termination criteria functions
    # Default to fully sequential terminating after a fixed number of iterations if not specified
    do_branch = cfg.do_branch if "do_branch" in cfg else fully_sequential
    assert "do_termination" in cfg or "num_iterations" in cfg, "Either do_termination or num_iterations must be defined in the config"
    do_termination = cfg.do_termination if "do_termination" in cfg else partial(fixed_iterations, cfg.num_iterations)

    # TODO: properly configure instatiation of these
    # Get the configurator, runner, and submitit_executor for branching
    submitit_executor = cfg.submitit_executor
    configurator = cfg.configurator
    runner = cfg.runner

    # Initialize important variables
    iterations = 0
    statistics = {}
    terminated = False
    sequential = do_branch(iterations, statistics)

    while not terminated:
        # Sequential: no branching, lr from the config
        if sequential:
            iterations += 1
            new_statistics = do_step(iterations, statistics)
            statistics = update_statistics(statistics, new_statistics)
            sequential = do_branch(iterations, statistics)
        # Branch: runner controls the branching path, configurator controls the lr
        else:
            # Configure slurm for the next jobs
            submitit_config = runner.get_submitit_config()
            submitit_executor.update_parameters(submitit_config)

            # Get arguments for the step and execute
            step_args = runner.get_step_args(configurator, statistics)

            running_jobs = []
            with submitit_executor.batch():
                for args in step_args:
                    job = submitit_executor.submit(do_step, **args)
                    running_jobs.append(job)

            # Collect results
            results = []
            for job in running_jobs:
                results.append(job.result())
        
            # Aggregate results
            new_statistics = aggregate_statistics(results)
            statistics = update_statistics(statistics, new_statistics)

        # Check termination
        terminated = do_termination(iterations, statistics)

    return statistics

def aggregate_statistics(results):
    statistics = {}
    for k in results[0].keys():
        statistics[k] = np.mean([r[k] for r in results])
        statistics[k + '_std'] = np.std([r[k] for r in results])
        statistics[k + '_min'] = np.min([r[k] for r in results])
        statistics[k + '_max'] = np.max([r[k] for r in results])
        for i, r in enumerate(results):
            statistics[f'{k}_{i}'] = r[k]
    return statistics

def update_statistics(old_statistics, new_statistics):
    updated_statistics = {}
    updated_statistics.update(new_statistics)
    updated_statistics["loss_history"] = old_statistics.get("loss_history", []) + [new_statistics["loss"]]
    updated_statistics["lr_history"] = old_statistics.get("lr_history", []) + [new_statistics["lr"]]
    return updated_statistics
