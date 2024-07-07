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
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    Integer,
)
from smac import Callback, Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband

from _hpo_transformer import dask_wrapper

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
        name="transformer",
        seed=seed,
        space={
            "pe_type": Categorical("pe_type", ["RoPE", "APE", "ALiBi"]),
            # "norm_type": Categorical("norm_type", ["RMSNorm", "LayerNorm"]),
            "rope_theta": Float("rope_theta", bounds=(500, 200_000)),
            # "loss": Categorical("loss", ["MSE", "MAE"]),
            "reduction": Categorical("reduction", ["sum", "mean"]),
            "dim_model": Categorical(
                "dim_model", [64, 128, 256, 384, 512, 768], ordered=True
            ),
            "n_heads": Categorical(
                "n_heads",
                [2, 4, 8, 12, 16, 32, 64, 128, 256, 384, 512],
                ordered=True,
            ),
            "seq_len": Categorical("seq_len", [256, 512, 1024, 2048], ordered=True),
            "n_layer": Integer("n_layer", bounds=(8, 25)),
            # "bias": Categorical("bias", [True, False], default=False),
            # "learning_rate": Float(
            #    "learning_rate",
            #    bounds=(1e-5, 1e-2),
            #    # log=True,
            #    default=1.5e-3,
            #    distribution=Normal(mu=5e-3, sigma=2),
            # ),
        },
    )

    cs.add_condition(EqualsCondition(cs["rope_theta"], cs["pe_type"], "RoPE"))

    # Function to find the forbidden heads for a given dim_model.
    def forbidden_heads_for_dim_model(dim_model, n_heads):
        return [head for head in n_heads if head >= dim_model or dim_model % head != 0]

    # Creating all forbidden clauses.
    forbidden_clauses = []
    for dim_model in cs["dim_model"].sequence:
        forbidden_heads = forbidden_heads_for_dim_model(
            dim_model, cs["n_heads"].sequence
        )
        if forbidden_heads:
            forbidden_dim_clause = ForbiddenEqualsClause(cs["dim_model"], dim_model)
            forbidden_heads_clause = ForbiddenInClause(cs["n_heads"], forbidden_heads)
            forbidden_clauses.append(
                ForbiddenAndConjunction(forbidden_dim_clause, forbidden_heads_clause)
            )

    cs.add_forbidden_clauses(forbidden_clauses)

    forbidden_dim_flash_attn = ForbiddenEqualsClause(cs["dim_model"], 768)
    forbidden_head_flash_attn = ForbiddenEqualsClause(cs["n_heads"], 2)
    forbidden_flash_attn = ForbiddenAndConjunction(
        forbidden_dim_flash_attn, forbidden_head_flash_attn
    )
    cs.add_forbidden_clauses([forbidden_flash_attn])

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace=cs,
        name="transformer_30",
        output_directory=Path(f"{Path.cwd()}/hpo"),
        deterministic=True,
        n_trials=200,
        # termination_cost_threshold=0.01,
        min_budget=333,
        max_budget=999,
        n_workers=4,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    # Create our intensifier
    intensifier = Hyperband(scenario, eta=3, incumbent_selection="highest_budget")

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

