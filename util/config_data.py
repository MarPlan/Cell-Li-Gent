from dataclasses import dataclass, field
from typing import Dict

import h5py
import numpy as np


@dataclass
class BatteryDatasheet:
    I_terminal: Dict[str, int] = field(
        default_factory=lambda: {
            "long_max": 360,  # [A]
            "long_min": -540,  # [A/5s]
            "short_max": 500,  # [A]
            "short_min": -700,  # [A]
            "short_time": 20,  # [s]
        }
    )
    U_terminal: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 4.3,  # [V]
            "min": 2.4,  # [V]
        }
    )
    T_surf: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 273.15 + 45,  # [K]
            "min": 273.15 + 10,  # [K]
        }
    )
    capa: Dict[str, float] = field(
        default_factory=lambda: {
            "SoH_100": 60,  # [Ah]
        }
    )
    # Optional for additional information
    c_rate: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 6,  # [.]
            "min": -9,  # [.]
        }
    )


def scale_data():
    # create scaler according to successfully tested datasheet params, scale dataset, save scaler object with ['DFN_synthetic_scaled'] signature
    # remember for complex scaling we have to seperate Re and Im, basically the output needs 2dim or a tuple
    pass


def create_dummy_dataset():
    # dt = 1.0
    params = BatteryDatasheet()
    n_series = 16
    seq_len = 1024
    input_params = ["I_terminal [A]", "U_terminal [V]", "T_surf [K]"]
    output_params = ["SoC [.]", "T_core [K]", "OCV [V]"]

    # Define the data structure to hold inputs and outputs
    data = np.empty((n_series, seq_len, len(input_params) + len(output_params)))

    # Assign ranges for inputs based on datasheet
    data[:, :, 0] = np.random.uniform(
        params.I_terminal["long_min"],
        params.I_terminal["long_max"],
        size=(n_series, seq_len),
    )  # Current [A]
    data[:, :, 1] = np.random.uniform(
        params.U_terminal["min"], params.U_terminal["max"], size=(n_series, seq_len)
    )  # Voltage [V]
    data[:, :, 2] = np.random.uniform(
        params.T_surf["min"], params.T_surf["max"], size=(n_series, seq_len)
    )  # Surface Temp [K]

    # Simulate simplistic values for output parameters for demonstration
    data[:, :, 3] = np.random.uniform(
        0.0, 1.0, size=(n_series, seq_len)
    )  # State of Charge [.]
    data[:, :, 4] = np.random.uniform(
        params.T_surf["min"], params.T_surf["max"], size=(n_series, seq_len)
    )  # Core Temp [K]
    data[:, :, 5] = np.random.uniform(
        params.U_terminal["min"], params.U_terminal["max"], size=(n_series, seq_len)
    )  # Open Circuit Voltage [V]

    # Open HDF5 file to write data
    with h5py.File("data/train/dummy_battery_data.h5", "w") as file:
        # Create dataset
        dataset = file.create_dataset("random", data=data)
        # Add attributes to describe each dimension
        dataset.attrs[
            "info"
        ] = """Dataset contains synthetic battery data from a DFN simulation
        dim[0] = n_series (number of series)
        dim[1] = seq_len: dt = 1s (sequence length)
        dim[2] = inputs : I_terminal [A], U_terminal [V], T_surface [K],
                 outputs: SoC [.]       , T_core [K]    , OCV [V]"""
        dataset.attrs["dim_2"] = (
            "Inp_I_terminal [A]",
            "Inp_U_terminal [V]",
            "Inp_T_surface [K]",
            "Out_SoC [.]",
            "Out_T_core [K]",
            "Out_OCV [V]",
        )


if __name__ == "__main__":
    create_dummy_dataset()
