import matplotlib.pyplot as plt
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
        # T_surf = variables["Volume-averaged surface temperature [K]"]
        # T_dict = {"Volume-averaged surface temperature [K]": T_surf}
        variables.update(self._get_standard_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature [K]"]
        Q_vol_av = variables["Volume-averaged total heating [W.m-3]"]
        T_surf = variables["Volume-averaged surface temperature [K]"]

        # Newton cooling, accounting for surface area to volume ratio
        h_fraction = 100
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


class LumpedSurface(pybamm.thermal.base_thermal.BaseThermal):
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
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T_surf = variables["Volume-averaged surface temperature [K]"]
        # T_vol_av = variables["Volume-averaged cell temperature [K]"]
        Q_vol_av = variables["Volume-averaged total heating [W.m-3]"]
        T_amb = variables["Volume-averaged ambient temperature [K]"]

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


if __name__ == "__main__":
    t = np.linspace(0, 900, 60)
    sin_t = 9.5 * np.sin(2 * np.pi * t)
    drive_cycle_power = np.column_stack([t, sin_t])

    parameter_values = pybamm.ParameterValues("Chen2020")
    options = {"cell geometry": "arbitrary"}
    model = pybamm.lithium_ion.DFN(options, build=False)

    model.submodels["thermal"] = Lumped(model.param, options)
    model.submodels["thermal_surf"] = LumpedSurface(model.param, options)
    model.build_model()

    experiment = pybamm.Experiment([pybamm.step.current(drive_cycle_power)])
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    solution = sim.solve()
    outputs = [
        "Current [A]",
        "Terminal voltage [V]",
        "X-averaged cell temperature [C]",
        "Volume-averaged surface temperature [C]",
    ]
    pybamm.dynamic_plot(solution, outputs)
    tt = 5
