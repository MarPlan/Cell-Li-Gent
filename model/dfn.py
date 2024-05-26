import numpy as np
import pybamm

t = np.linspace(0, 1, 60)
sin_t = 0.5 * np.sin(2 * np.pi * t)
drive_cycle_power = np.column_stack([t, sin_t])

parameter_values = pybamm.ParameterValues("Chen2020")
options = {"cell geometry": "arbitrary", "thermal": "lumped"}
model = pybamm.lithium_ion.DFN(options)
experiment = pybamm.Experiment([pybamm.step.current(drive_cycle_power)])
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
solution = sim.solve()
outputs = ["Current [A]", "Terminal voltage [V]"]
pybamm.dynamic_plot(solution, outputs)
tt = 5
