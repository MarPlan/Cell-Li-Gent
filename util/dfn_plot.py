import os

import matplotlib.pyplot as plt
import numpy as np
import pybamm

from config_data import BatteryDatasheet

if __name__ == "__main__":
    # parameter_values = pybamm.ParameterValues("Chen2020")
    # options = {"cell geometry": "arbitrary", "thermal": "lumped"}
    # model = pybamm.lithium_ion.DFN(options)
    # initialize = pybamm.Experiment(
    #     [
    #         (
    #             "Discharge at 1C until 2.5V",
    #             "Rest for 360 second",
    #             "Charge at 1C until 4.2V",
    #             "Hold at 4.2V until C/20",
    #         )
    #     ]
    #     * 5,
    # )
    # sim = pybamm.Simulation(
    #     model, parameter_values=parameter_values, experiment=initialize
    # )
    # solution = sim.solve()
    # solution.save("model_init.pkl")

    specs = BatteryDatasheet()
    parameter_values = pybamm.ParameterValues("Chen2020")
    options = {"cell geometry": "arbitrary", "thermal": "lumped"}
    model = pybamm.lithium_ion.DFN(options)
    curr = np.load("data/current/test_dfn.npy", allow_pickle=True)[2, :239_000]
    t = np.arange(0, curr.shape[0], specs.dt)
    cycle = np.column_stack([t, curr])
    # model_init = pybamm.load("model_init.pkl")
    # model.set_initial_conditions_from(model_init, inplace=True)

    # create interpolant
    current_interpolant = pybamm.Interpolant(cycle[:, 0], cycle[:, 1], pybamm.t)
    # set drive cycle
    parameter_values["Current function [A]"] = current_interpolant

    # experiment = pybamm.Experiment([pybamm.step.current(cycle)])
    # TODO: TQMD for simulation
    sim = pybamm.Simulation(
        model,
        parameter_values=parameter_values,
        solver=pybamm.CasadiSolver(
            mode="fast", extra_options_setup={"max_step_size": 1}
        ),  # experiment=experiment
    )
    solution = sim.solve()
    solution.save('profile_2.pkl')

    outputs = [
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [C]",
    ]
    pybamm.dynamic_plot(solution, outputs)
    tt = 5
