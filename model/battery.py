import copy
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pybamm


class Lumped(pybamm.thermal.base_thermal.BaseThermal):
    """
    Class for lumped thermal submodel. For more information see :footcite:t:`Timms2021`
    and :footcite:t:`Marquis2020`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)
        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        T_vol_av = pybamm.Variable(
            "Volume-averaged cell temperature [K]",
            scale=self.param.T_ref,
            print_name="T_av",
        )
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])

        T_surf = pybamm.Variable(
            "Volume-averaged surface temperature [K]",
            scale=self.param.T_ref,
            print_name="T_surf",
        )

        T_dict = {
            "volume-averaged surface": T_surf,
            "negative current collector": T_x_av,
            "positive current collector": T_x_av,
            "x-averaged cell": T_x_av,
            "volume-averaged cell": T_vol_av,
        }
        for domain in ["negative electrode", "separator", "positive electrode"]:
            T_dict[domain] = pybamm.PrimaryBroadcast(T_x_av, domain)

        variables = self._get_standard_fundamental_variables(T_dict)
        return variables

    def get_coupled_variables(self, variables):
        T_surf = variables["Volume-averaged surface temperature [K]"]
        T_dict = {"Volume-averaged surface temperature [K]": T_surf}
        variables.update(self._get_standard_coupled_variables(variables))
        variables.update(T_dict)
        return variables

    def set_rhs(self, variables):
        """
        Simple attempt for assuming only thermal conduction between cell and surface:
        in theory: R_cond ~ d/(kA) and R_conv ~ 1/(hA)
        here h_fraction represents only a very simplified case
        """
        T_vol_av = variables["Volume-averaged cell temperature [K]"]
        Q_vol_av = variables["Volume-averaged total heating [W.m-3]"]
        T_surf = variables["Volume-averaged surface temperature [K]"]

        # Newton cooling, accounting for surface area to volume ratio
        h_fraction = 3
        cell_surface_area = self.param.A_cooling
        cell_volume = self.param.V_cell
        Q_cool_vol_av = (
            -self.param.h_total
            / h_fraction
            * (T_vol_av - T_surf)
            * cell_surface_area
            / cell_volume
        )

        self.rhs = {
            T_vol_av: (Q_vol_av + Q_cool_vol_av) / self.param.rho_c_p_eff(T_vol_av)
        }

    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature [K]"]
        self.initial_conditions = {T_vol_av: self.param.T_init}


class LumpedSurface(Lumped):
    """
    Not coupling heat sources from core and surface directly to realize a different
    time constant, not very realistic but perhaps good enough for a fast evaluation
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)
        pybamm.citations.register("Timms2021")

    def set_rhs(self, variables):
        T_surf = variables["Volume-averaged surface temperature [K]"]
        Q_vol_av = variables["Volume-averaged total heating [W.m-3]"]
        T_amb = variables["Volume-averaged ambient temperature [K]"]
        T_vol_av = variables["Volume-averaged cell temperature [K]"]

        # Newton cooling, accounting for surface area to volume ratio
        cell_surface_area = self.param.A_cooling
        cell_volume = self.param.V_cell

        Q_cool_vol_av = (
            -self.param.h_total * (T_surf - T_amb) * cell_surface_area / cell_volume
        )

        self.rhs = {T_surf: (Q_vol_av + Q_cool_vol_av) / self.param.rho_c_p_eff(T_surf)}

    def set_initial_conditions(self, variables):
        T_surf = variables["Volume-averaged surface temperature [K]"]
        self.initial_conditions = {T_surf: self.param.T_init}


def solve_simulation(idx):
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values["Total heat transfer coefficient [W.m-2.K-1]"] = 2
    options = {"cell geometry": "arbitrary"}
    # model = pybamm.lithium_ion.DFN(options, build=False)
    model = pybamm.lithium_ion.SPMe(options, build=False)
    model.submodels["thermal"] = Lumped(model.param, options)
    model.submodels["thermal_surf"] = LumpedSurface(model.param, options)
    model.build_model()

    t = np.arange(0, current[idx].shape[0], 1)
    cycle = np.column_stack([t, current[idx]])
    current_interpolant = pybamm.Interpolant(cycle[:, 0], cycle[:, 1], pybamm.t)
    l_parameter_values = copy.deepcopy(parameter_values)
    l_parameter_values["Current function [A]"] = current_interpolant
    print(f"Running Process:{os.getpid()}, profile:{idx}/{current.shape[0]}")
    sim = pybamm.Simulation(
        model,
        parameter_values=l_parameter_values,
        solver=pybamm.CasadiSolver(
            mode="fast", extra_options_setup={"max_step_size": 1}
        ),
    )

    solution = sim.solve()
    outputs = [
        "Current [A]",
        "Terminal voltage [V]",
        "Volume-averaged surface temperature [C]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [C]",
        "Battery open-circuit voltage [V]",
    ]

    # Define the data structure to hold inputs and outputs
    capa = 5
    data = np.array(
        [
            solution[key].entries
            if key != "Discharge capacity [A.h]"
            else solution[key].entries / capa
            for key in outputs
        ]
    )

    np.save(f"data/train/spme_{idx}.npy", data.T)
    print(
        f"Successfully Saved: Process:{os.getpid()}, profile:{idx}/{current.shape[0]}"
    )


def initialize_globals(_current):
    global current
    current = _current


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(".."))
    from util.config_data import BatteryDatasheet

    os.chdir("..")
    specs = BatteryDatasheet()
    current = np.load("data/current/current_profiles.npy")

    with ProcessPoolExecutor(
        max_workers=3,
        initializer=initialize_globals,
        initargs=(current,),
    ) as executor:
        results = list(executor.map(solve_simulation, range(291, current.shape[0])))

