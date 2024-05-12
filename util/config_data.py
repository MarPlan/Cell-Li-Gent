import os
from dataclasses import dataclass, field
from typing import Dict

import h5py
import numpy as np


@dataclass
class BatteryDatasheet:
    # Provide all data ranges for input and output!
    # These get tested and scaled accordingly
    # If not specified here -> min/max values for scaling from
    # dataset and no testing for unspecified values
    I_terminal: Dict[str, int] = field(
        default_factory=lambda: {
            "max": 360,  # [A] continous
            "min": -540,  # [A] continous
            "short_max": 500,  # [A/s]
            "short_min": -700,  # [A/s]
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
            "max": 60,  # [Ah] @ SoH 100
            "min": 0,  # [Ah] @ SoH 100
        }
    )
    # Optional for additional information, unused
    c_rate: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 6,  # [.]
            "min": -9,  # [.]
        }
    )


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