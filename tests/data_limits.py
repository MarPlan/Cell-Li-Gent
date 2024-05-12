import re

import numpy as np

from util.config_data import BatteryDatasheet


def verify_data_limits(dataset):
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
            max_limit = limits.I_terminal["max"]
            min_limit = limits.I_terminal["min"]
            if (data > max_limit).any() or (data < min_limit).any():
                raise ValueError(
                    f"I_terminal values should be between {min_limit} and {max_limit}"
                )
            verified_values["I_terminal_max"] = data.max()
            verified_values["I_terminal_min"] = data.min()

            # Short term checks
            window_size = limits.I_terminal["short_time"] // dt
            short_max_limit = limits.I_terminal["short_max"]
            short_min_limit = limits.I_terminal["short_min"]
            for start in range(data.shape[1] - window_size + 1):
                window_data = data[:, start : start + window_size]
                if (window_data.max(axis=1) > short_max_limit).any() or (
                    window_data.min(axis=1) < short_min_limit
                ).any():
                    raise ValueError(
                        f"I_terminal short-term values should be between "
                        f"{short_min_limit} and {short_max_limit} within any "
                        f"{limits.I_terminal['short_time']}s window"
                    )
            verified_values["I_terminal_short_max"] = np.max(
                [
                    data[:, i : i + window_size].max()
                    for i in range(data.shape[1] - window_size + 1)
                ]
            )
            verified_values["I_terminal_short_min"] = np.min(
                [
                    data[:, i : i + window_size].min()
                    for i in range(data.shape[1] - window_size + 1)
                ]
            )

            # Capacity check
            charge = np.trapz(data, dx=dt, axis=1) / 60 / 60 # convert to [Ah]
            if (charge > limits.capa["max"]).any() or (
                charge < -limits.capa["max"]
            ).any():
                raise ValueError("Charge capacity outside of allowable range")
            verified_values["capa_max"] = charge.max()
            verified_values["capa_min"] = charge.min()

        elif param == "U_terminal":
            max_limit = limits.U_terminal["max"]
            min_limit = limits.U_terminal["min"]
            if (data > max_limit).any() or (data < min_limit).any():
                raise ValueError(
                    f"U_terminal values should be between {min_limit} and {max_limit}"
                )
            verified_values["U_terminal_max"] = data.max()
            verified_values["U_terminal_min"] = data.min()

        elif param == "T_surface":
            max_limit = limits.T_surf["max"]
            min_limit = limits.T_surf["min"]
            if (data > max_limit).any() or (data < min_limit).any():
                raise ValueError(
                    f"T_surface values should be between {min_limit} and {max_limit}"
                )
            verified_values["T_surface_max"] = data.max()
            verified_values["T_surface_min"] = data.min()

    dataset.attrs["verified_values"] = str(verified_values)
