import h5py
import matplotlib.pyplot as plt

from tests.data_limits import verify_data_limits
from util.config_data import scale_data

if __name__ == "__main__":
    file_path = "data/train/dummy_battery_data.h5"
    dataset_name = "random"
    with h5py.File(file_path, "r+") as file:
        dataset = file[dataset_name]
        # verify_data_limits(dataset)
        # scale_data(file_path, dataset_name)
        x=5
        # plt.plot(file["random"][0,0,0])
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Random', color=color)
        ax1.plot(file["random"][0, :, 0], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Random Scaled', color=color)
        ax2.plot(file["random_scaled"][0, :, 0],'--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
