import os
from dataclasses import dataclass, field
from typing import Dict

import h5py
import numpy as np


@dataclass
class BatteryDatasheet:
    # INFO: CHEN2020
    # LG INR21700 M50:
    # https://www.dnkpower.com/wp-content/uploads/2019/02/LG-INR21700-M50-Datasheet.pdf
    I_terminal: Dict[str, float] = field(
        default_factory=lambda: {
            "chg": -3.395,  # [A] continous
            "dchg": 7.275,  # [A] continous
            "short_chg": -3.395,  # [A/s]
            "short_dchg": 80/4.2,  # 19.04 [A/s]
            "short_time": 10,  # [s]
            # Parameter for current profile generation
            "soc_crit_chg": -1.455,
            "soc_crit_dchg": 0.970,
        }
    )
    U_terminal: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 4.2,  # [V]
            "min": 2.5,  # [V]
        }
    )
    T_surf: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 273.15 + 60,  # [K]
            "min": 273.15 + 25,  # [K]
        }
    )
    capa: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 5.0,  # [Ah] @ SoH 100
            "min": 0,  # [Ah] @ SoH 100
            # Parameter for current profile generation
            "soc_crit_max": 0.8,
            "soc_crit_min": 0.2,
            "soc_max": 0.98,
            "soc_min": 0.01,
        }
    )
    # Parameter for current profile generation
    dt: int = 1
    # Optional for additional information, unused
    c_rate: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 6,  # [.]
            "min": -9,  # [.]
        }
    )

    def __post_init__(self):
        self.cycle_time = (
            int(self.capa["max"]) * 20 * 3600
        )  # C/20 as equivalent duration [s]
        self.seq_len = self.cycle_time // self.dt


def scale_data(file_path, dataset_name):
    """Scale data according to the specification if in or output is not
    specified in the dataclass the min and max gets determined from the
    actual data"""
    datasheet = BatteryDatasheet()

    with h5py.File(file_path, "r+") as file:
        dataset = file[dataset_name]
        data = dataset[:]

        feature_names = dataset.attrs["dim_2"]
        min_values = []
        max_values = []

        # Scaling data based on feature_names from dataset.attrs
        scaled_data = np.empty_like(data)
        for i, name in enumerate(feature_names):
            clean_name = (
                name.split(" ")[0][4:] if "Inp_" in name else name.split(" ")[0][4:]
            )  # Removes 'Inp_'/'Out_' and unit
            if clean_name in datasheet.__dict__:
                min_val = datasheet.__dict__[clean_name]["min"]
                max_val = datasheet.__dict__[clean_name]["max"]
            else:
                min_val = data[:, :, i].min()
                max_val = data[:, :, i].max()
            min_values.append(min_val)
            max_values.append(max_val)
            scaled_data[:, :, i] = (data[:, :, i] - min_val) / (max_val - min_val)

        # Add the scaled dataset
        scaled_dataset = file.create_dataset(dataset_name + "_scaled", data=scaled_data)
        scaled_dataset.attrs["min_values"] = min_values
        scaled_dataset.attrs["max_values"] = max_values

    print("Data succesfully scaled and stored")


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
        params.I_terminal["min"],
        params.I_terminal["max"],
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
        params.T_surf["min"] + 1, params.T_surf["max"] + 5, size=(n_series, seq_len)
    )  # Core Temp [K]
    data[:, :, 5] = np.random.uniform(
        params.U_terminal["min"], params.U_terminal["max"], size=(n_series, seq_len)
    )  # Open Circuit Voltage [V]

    data_dir = os.path.abspath("../data/train/dummy_battery_data.h5")
    # Open HDF5 file to write data
    with h5py.File(data_dir, "w") as file:
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
    print("Dummy data created succesfully")


if __name__ == "__main__":
    create_dummy_dataset()
