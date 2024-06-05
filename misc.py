import glob

import matplotlib.pyplot as plt
import numpy as np

from tests.data_limits import verify_dataset_limits
from util.config_data import BatteryDatasheet, scale_data


def compare_dfn_spm():
    dfn_0 = np.load("data/train/dfn_1.npy")
    # dfn_1 = np.load("data/train/dfn_1.npy")
    # dfn_2 = np.load("data/train/dfn_2.npy")

    spme_0 = np.load("data/train/spme_1.npy")
    # spme_1 = np.load("data/train/spme_1.npy")
    # spme_2 = np.load("data/train/spme_2.npy")

    fig = plt.figure()
    ax = fig.subplots(3, 1)

    ax[0].plot(dfn_0[:, 0])
    ax[0].plot(spme_0[:, 0], "--")

    ax[1].plot(dfn_0[:, 1])
    ax[1].plot(spme_0[:, 1], "--")

    ax[1].plot(dfn_0[:, -1])
    ax[1].plot(spme_0[:, -1], "--")

    ax[2].plot(dfn_0[:, 2])
    ax[2].plot(spme_0[:, 2], "--")

    ax[2].plot(dfn_0[:, -2])
    ax[2].plot(spme_0[:, -2], "--")
    plt.show()


def stack_spme_output_to_dataset():
    # Path to your directory
    dir_path = "data/train/"

    # Find all files matching the pattern spme_x.npy
    file_pattern = dir_path + "/spme_*.npy"
    file_list = glob.glob(file_pattern)

    # Load each file and collect them in a list
    arrays_list = [np.load(file) for file in file_list]

    # Stack arrays along the first dimension
    data = np.stack(arrays_list, axis=0)

    import os

    import h5py

    data_dir = os.path.abspath("data/train/battery_data.h5")
    # Open HDF5 file to write data
    with h5py.File(data_dir, "a") as file:
        print("Creating new dataset")
        dataset = file.create_dataset(
            "spme_training",
            data=data,
            maxshape=(None, data.shape[1], data.shape[2]),
            dtype=float,
        )

        # Add attributes to describe each dimension
        dataset.attrs[
            "info"
        ] = """Dataset contains synthetic battery data from a DFN simulation
        dim[0] = n_series (number of series)
        dim[1] = seq_len: dt = 1s (sequence length)
        dim[2] = inputs : I_terminal [A], U_terminal [V], T_surface [C],
                 outputs: SoC [.]       , T_core [C]    , OCV [V]"""
        dataset.attrs["dim_2"] = (
            "Inp_I_terminal [A]",
            "Inp_U_terminal [V]",
            "Inp_T_surface [C]",
            "Out_SoC [.]",
            "Out_T_core [C]",
            "Out_OCV [V]",
        )


if __name__ == "__main__":
    # compare_dfn_spm()
    stack_spme_output_to_dataset()
