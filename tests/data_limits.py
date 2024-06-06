import re

import h5py
import numpy as np
from util.config_data import BatteryDatasheet


def check_crit_current(
    current,
    dt,
    capa,
    soc_start,
    soc_crit_min,
    soc_crit_max,
    curr_crit_min,
    curr_crit_max,
):
    # Calculate the SoC values
    soc = soc_start - (np.cumsum(current, axis=1) * dt / 3600 / capa)
    # Find indices where SoC is outside the critical range
    crit_soc_mask = (soc < soc_crit_min) | (soc > soc_crit_max)
    # Find corresponding current values
    crit_current = current[crit_soc_mask]
    # Check if any critical current values violate the constraints
    if np.any((crit_current < curr_crit_min) & (crit_current > curr_crit_max)):
        raise ValueError(
            f"Critical current limit violation: Expected current to be "
            f"< {curr_crit_min} or > {curr_crit_max} "
            f"when SoC is outside the range [{soc_crit_min}, {soc_crit_max}]."
        )
    print("Passed: Critical current check")


def check_bounds_current(current, max_limit, min_limit):
    if (current > max_limit).any() or (current < min_limit).any():
        raise ValueError(
            f"I_terminal values should be between {min_limit} and {max_limit} "
            f"but is min={current.min()}, max={current.max()}"
        )
    print(f"Passed: Current bounds min={current.min()}, max={current.max()}")


def check_short_current(current, short_max_limit, short_min_limit, short_time, dt):
    window_size = short_time // dt
    for start in range(current.shape[1] - window_size + 1):
        window_data = current[:, start : start + window_size]

        if (window_data.max(axis=1) > short_max_limit).any() or (
            window_data.min(axis=1) < short_min_limit
        ).any():
            raise ValueError(
                f"I_terminal short-term values should be between "
                f"{short_min_limit} and {short_max_limit} within any "
                f"{short_time}s window "
                f"but is min={current.min()}, max={current.max()}"
            )
    print("Passed: Short current")


def check_soc(
    current,
    dt,
    capa,
    soc_start,
    capa_soc_max,
    capa_soc_min,
):
    charge = soc_start - (np.cumsum(current, axis=1) * dt / 3600 / capa)
    # convert to [Ah]
    if (charge > capa_soc_max).any() or (charge < capa_soc_min).any():
        raise ValueError(
            f"Charge capacity outside of allowable range "
            f"min={charge.min()}, max={charge.max()}"
        )
    print(f"Passed: SoC min={charge.min()}, max={charge.max()}")
    return charge


def check_bounds_voltage(voltage, max_limit, min_limit):
    if (voltage > max_limit).any() or (voltage < min_limit).any():
        raise ValueError(
            f"U_terminal values should be between {min_limit} and {max_limit}"
        )
    print(f"Passed: Voltage bounds min={voltage.min()}, max={voltage.max()}")


def check_bounds_temp(temp, max_limit, min_limit):
    if (temp > max_limit).any() or (temp < min_limit).any():
        raise ValueError(
            f"T_surface values should be between {min_limit} and {max_limit}"
        )
    print(f"Passed: Temperature bounds min={temp.min()}, max={temp.max()}")


def verify_dataset_limits(data_file, dataset):
    with h5py.File(data_file, "r+") as file:
        dataset = file[dataset]  # Access the specified dataset
        # Currently only inputs get checked
        limits = BatteryDatasheet()
        inp_params = [
            param.split(" ")[0][4:] if "Inp_" in param else None
            for param in dataset.attrs["dim_2"]
        ]

        # Extracting dt from the seq_len attribute:
        dt_match = re.search(r"dt\s*=\s*(\d+)s", dataset.attrs["info"])
        dt = int(dt_match.group(1)) if dt_match else 1  # default to 1 if not found
        verified_values = {}
        for index, param in enumerate(inp_params):
            if param is None:
                continue
            data = dataset[:, :, index]
            if param == "I_terminal":
                max_limit = limits.I_terminal["short_chg"]
                min_limit = limits.I_terminal["short_dchg"]
                check_bounds_current(data, min_limit, max_limit)
                verified_values["I_terminal_max"] = data.max()
                verified_values["I_terminal_min"] = data.min()
                # Short term checks
                short_max_limit = limits.I_terminal["short_chg"]
                short_min_limit = limits.I_terminal["short_dchg"]
                short_time = limits.I_terminal["short_time"]
                check_short_current(
                    data, short_min_limit, short_max_limit, short_time, dt
                )
                soc_start = 1
                check_crit_current(
                    data,
                    dt,
                    limits.capa["max"],
                    soc_start,
                    limits.capa["soc_crit_min"],
                    limits.capa["soc_crit_max"],
                    limits.I_terminal["soc_crit_chg"],
                    limits.I_terminal["soc_crit_dchg"],
                )
                # Capacity check
                charge = check_soc(
                    data,
                    dt,
                    limits.capa["max"],
                    soc_start,
                    max(limits.capa["soc_max"], soc_start)*1.001,
                    limits.capa["soc_min"]*0.999,
                )
                verified_values["SoC_max"] = charge.max()
                verified_values["SoC_min"] = charge.min()
            elif param == "U_terminal":
                max_limit = limits.U_terminal["max"]
                min_limit = limits.U_terminal["min"]
                check_bounds_voltage(data, max_limit, min_limit)
                verified_values["U_terminal_max"] = data.max()
                verified_values["U_terminal_min"] = data.min()
            elif param == "T_surface":
                max_limit = limits.T_surf["max"]
                min_limit = limits.T_surf["min"]
                check_bounds_temp(data, max_limit, min_limit)
                verified_values["T_surface_max"] = data.max()
                verified_values["T_surface_min"] = data.min()
        dataset.attrs["verified_values"] = str(verified_values)
