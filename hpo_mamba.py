"""
Train for half an epoch -> should continue till end
Use max possible memory, catch mem error and reduce batch size, overall 500k, if too big loss inf
Keep warmup, min lr, scheduler fixed, linear decay to min lr within one epoch
Use auto regressive for predciton/test/objective loss, meaning only current external
"""

import gc
from pathlib import Path

import numpy as np
import torch
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Integer,
)
from smac import Callback, Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband

from _hpo_mamba import dask_wrapper

if __name__ == "__main__":
    gpu_list = [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        # "cuda:4",
        # "cuda:5",
        # "cuda:6",
        # "cuda:7",
    ]

    with open("gpu_list.txt", "w") as file:
        for gpu in gpu_list:
            file.write(f"{gpu}\n")

    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)

    cs = ConfigurationSpace(
        name="mamba",
        seed=seed,
        space={
            "d_intermediate": Categorical("d_intermediate", [0, 1], ordered=True),
            "dim_model": Categorical("dim_model", [256, 512, 768], ordered=True),
            "seq_len": Categorical(
                "seq_len", [512, 768, 1024, 1536, 2048], ordered=True
            ),
            "n_layer": Integer("n_layer", bounds=(6, 30)),
        },
    )

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace=cs,
        name="mamba",
        output_directory=Path(f"{Path.cwd()}/hpo"),
        deterministic=True,
        n_trials=200,
        # termination_cost_threshold=0.01,
        min_budget=100,
        max_budget=1000,
        n_workers=4,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    # Create our intensifier
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    class CustomCallback(Callback):
        def __init__(self) -> None:
            pass

        def on_iteration_start(self, smbo) -> None:
            # torch._dynamo.reset()
            # torch._C._cuda_clearCublasWorkspaces()
            gc.collect()
            torch.cuda.empty_cache()
            return None

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        dask_wrapper,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=False,
        logging_level=20,
        callbacks=[CustomCallback()],
    )

    # Let's optimize
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")
