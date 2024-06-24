import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tests.data_limits import verify_dataset_limits
from util.config_data import check_scale_rescale, rescale_data, scale_data


# Open the existing HDF5 file
def copy_dataset(data_dir= "data/train/battery_data.h5", dataset="spme_training_scaled"):
    with h5py.File(data_dir, "r") as file:
        original_dataset = file[dataset]
        # Create a new HDF5 file to store the copied dataset
        with h5py.File("debug_data.h5", "w") as new_file:
            # Create the new dataset with the same properties as the original
            new_dataset = new_file.create_dataset(
                "spme_training_scaled",
                shape=(30, original_dataset.shape[1], original_dataset.shape[2]),
                maxshape=(30, original_dataset.shape[1], original_dataset.shape[2]),
                dtype=original_dataset.dtype,
            )
            # Copy attributes from the original dataset to the new dataset
            for attr_name in ["info","dim_2","verified_values","min_values","max_values"]:
                new_dataset.attrs[attr_name] = original_dataset.attrs[attr_name]
            # Copy data in chunks to avoid using too much memory
            new_dataset[:] = original_dataset[:30]


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
    dir_path = "data/train/raw_spme"

    # Find all files matching the pattern spme_x.npy
    file_pattern = dir_path + "/spme_*.npy"
    file_list = glob.glob(file_pattern)
    file_list.sort()

    import os
    # Extract the numerical part from each filename
    # extracted_numbers = set()
    # for filename in file_list:
    #     # Extract the number from the filename
    #     number = int(os.path.splitext(os.path.basename(filename))[0].split("_")[1])
    #     extracted_numbers.add(number)
    #
    # # Expected range of numbers
    # expected_numbers = set(range(3000))
    #
    # # Find the missing number
    # missing_numbers = expected_numbers - extracted_numbers
    #
    # if missing_numbers:
    #     print(f"The missing file(s): {sorted(missing_numbers)}")
    # else:
    #     print("No files are missing. All files are present.")

    initial_data = np.load(file_list[0])
    seq_len, dims = initial_data.shape[0], initial_data.shape[1]

    import h5py

    data_dir = os.path.abspath("data/train/battery_data.h5")
    # Open HDF5 file to write data
    with h5py.File(data_dir, "w") as file:
        print("Creating new dataset")
        dataset = file.create_dataset(
            "spme_training",
            data=initial_data,
            shape=(1, seq_len, dims),
            maxshape=(None, seq_len, dims),
            dtype=initial_data.dtype,
        )

        # Append data from each file to the dataset
        dataset.resize(len(file_list), axis=0)
        for i, file_name in enumerate(file_list):
            data = np.load(file_name)
            dataset[i] = data

        # Add attributes to describe each dimension
        dataset.attrs[
            "info"
        ] = """Dataset contains synthetic battery data from a SPMe simulation
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
    with h5py.File("debug_data.h5", "r") as file:
        original_dataset = file["spme_training_scaled"]
    copy_dataset()
    # compare_dfn_spm()
    # stack_spme_output_to_dataset()
    # data_file = "data/train/battery_data.h5"
    # dataset = "spme_training"
    # verify_dataset_limits(data_file=data_file, dataset=dataset)
    # scale_data(file_path=data_file, dataset_name=dataset)
    # rescale_data(file_path=data_file, dataset_name=dataset + "_scaled")
    # check_scale_rescale(file_path=data_file, dataset_name=dataset)
