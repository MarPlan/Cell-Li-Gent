import os
from dataclasses import dataclass, field
from typing import Dict

import h5py
import matplotlib.pyplot as plt
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
            "short_dchg": 80 / 4.2,  # 19.04 [A/s]
            "short_time": 10,  # [s]
            # Parameter for current profile generation
            "soc_crit_chg": -1.455,
            "soc_crit_dchg": 0.970,
        }
    )
    U_terminal: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 4.25,  # [V]
            "min": 2.5,  # [V]
        }
    )
    T_surf: Dict[str, float] = field(
        default_factory=lambda: {
            "max": 60,  # [C]
            "min": -25,  # [C]
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


def rescale_data(file_path, dataset_name):
    """Rescale data back to the original values using stored min and max values."""

    with h5py.File(file_path, "r+") as file:
        scaled_dataset = file[dataset_name]
        feature_names = scaled_dataset.attrs["dim_2"].tolist()
        min_values = scaled_dataset.attrs["min_values"]
        max_values = scaled_dataset.attrs["max_values"]

        # Create a new dataset for rescaled data
        rescaled_dataset = file.create_dataset(
            dataset_name + "_rescaled", scaled_dataset.shape, dtype=scaled_dataset.dtype
        )

        # Rescale dataset slice by slice
        for slice_idx in range(scaled_dataset.shape[0]):
            scaled_slice = scaled_dataset[slice_idx, :, :]
            rescaled_slice = np.empty_like(scaled_slice)
            for i, _ in enumerate(feature_names):
                rescaled_slice[:, i] = (
                    scaled_slice[:, i] * (max_values[i] - min_values[i])
                ) + min_values[i]
            rescaled_dataset[slice_idx, :, :] = rescaled_slice

        # Copy attributes
        for attr in list(scaled_dataset.attrs.keys()):
            rescaled_dataset.attrs[attr] = scaled_dataset.attrs[attr]

    print("Data successfully rescaled and stored")


def scale_data(file_path, dataset_name):
    """Scale data according to the specification if in or output is not
    specified in the dataclass the min and max gets determined from the
    actual data"""

    datasheet = BatteryDatasheet()

    with h5py.File(file_path, "r+") as file:
        dataset = file[dataset_name]
        feature_names = dataset.attrs["dim_2"].tolist()

        min_values = []
        max_values = []
        scaled_dataset = file.create_dataset(
            dataset_name + "_scaled", dataset.shape, dtype=dataset.dtype
        )
        verified_values = dataset.attrs["verified_values"]

        verified_values = verified_values.strip('"')
        verified_values = eval(verified_values)

        for i, name in enumerate(feature_names):
            clean_name = name.split(" ")[0][4:]

            if clean_name in datasheet.__dict__:
                if clean_name == "I_terminal":  # Specific handling for I_terminal
                    min_val = datasheet.__dict__[clean_name]["short_chg"]
                    max_val = datasheet.__dict__[clean_name]["short_dchg"]
                else:
                    min_val = datasheet.__dict__[clean_name]["min"]
                    max_val = datasheet.__dict__[clean_name]["max"]
                print(f"{clean_name} scaled based on datasheet")

            elif any(clean_name in key for key in verified_values.keys()):
                keys = [key for key in verified_values.keys() if clean_name in key]
                min_val = verified_values[[key for key in keys if "min" in key][0]]
                max_val = verified_values[[key for key in keys if "max" in key][0]]
                print(f"{clean_name} scaled based on verified_values")

            else:
                min_val, max_val = float("inf"), float("-inf")
                for slice_idx in range(dataset.shape[0]):
                    slice_data = dataset[slice_idx, :, i]
                    min_val = min(min_val, slice_data.min())
                    max_val = max(max_val, slice_data.max())
                print(f"{clean_name} scaled based on dataset")

            min_values.append(min_val)
            max_values.append(max_val)

        # Scale dataset slice by slice
        for slice_idx in range(dataset.shape[0]):
            data_slice = dataset[slice_idx, :, :]
            scaled_slice = np.empty_like(data_slice)
            for i, _ in enumerate(feature_names):
                scaled_slice[:, i] = (data_slice[:, i] - min_values[i]) / (
                    max_values[i] - min_values[i]
                )
            scaled_dataset[slice_idx, :, :] = scaled_slice

        scaled_dataset.attrs["min_values"] = min_values
        scaled_dataset.attrs["max_values"] = max_values

        # Copy attributes
        for attr in list(dataset.attrs.keys()):
            scaled_dataset.attrs[attr] = dataset.attrs[attr]
    print("Data successfully scaled and stored")


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
        params.I_terminal["chg"],
        params.I_terminal["dchg"],
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
        ] = """Dataset contains synthetic battery data from a SPMe simulation
        dim[0] = n_series (number of series)
        dim[1] = seq_len: dt = 1s (sequence length)
        dim[2] = inputs : I_terminal [A], U_terminal [V], T_surface [K],
                 outputs: SoC [.]       , T_core [K]    , OCV [V]"""
        dataset.attrs["dim_2"] = (
            "Inp_I_terminal [A]",
            "Inp_U_terminal [V]",
            "Inp_T_surface [C]",
            "Out_SoC [.]",
            "Out_T_core [C]",
            "Out_OCV [V]",
        )
    print("Dummy data created succesfully")


def check_scale_rescale(file_path, dataset_name):
    with h5py.File(file_path, "r") as file:
        data_raw = file[dataset_name]
        data_scaled = file[dataset_name + "_scaled"]
        data_resacled = file[dataset_name + "_scaled" + "_rescaled"]

        plt.plot(data_raw[0, :, 0], label="raw")
        plt.plot(data_resacled[0, :, 0], "--", label="rescaled")

        v = data_raw[0, :, 0]
        curr_min, curr_max = data_scaled.attrs["min_values"][0], data_scaled.attrs["max_values"][0]
        val = (v - curr_min) / (curr_max - curr_min)
        plt.plot(val, label="scaled val")

        plt.plot(data_scaled[0, :, 0], "--", label="scaled")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import sys


    sys.path.append(os.path.abspath(".."))
    # os.chdir("..")
    from tests.data_limits import verify_dataset_limits
    from util.config_data import scale_data

    create_dummy_dataset()
    verify_dataset_limits(
        data_file="../data/train/dummy_battery_data.h5", dataset="random"
    )
    scale_data(file_path="../data/train/dummy_battery_data.h5", dataset_name="random")
    rescale_data(
        file_path="../data/train/dummy_battery_data.h5", dataset_name="random_scaled"
    )

    check_scale_rescale(
        file_path="../data/train/dummy_battery_data.h5", dataset_name="random"
    )
    t = 5
