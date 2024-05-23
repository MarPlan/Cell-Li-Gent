import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from tests.data_limits import verify_dataset_limits
from util.config_data import BatteryDatasheet, scale_data

if __name__ == "__main__":
    specs = BatteryDatasheet()
    curr_chg = specs.I_terminal["chg"]  # [A] continous
    curr_dchg = specs.I_terminal["dchg"]  # [A] continous
    curr_short_chg = specs.I_terminal["short_chg"]  # [A/s]
    curr_short_dchg = specs.I_terminal["short_dchg"]  # [A/s]
    curr_short_time = specs.I_terminal["short_time"]  # [s]
    curr_soc_crit_chg = specs.I_terminal["soc_crit_chg"]
    curr_soc_crit_dchg = specs.I_terminal["soc_crit_dchg"]

    capa = specs.capa["max"]  # [Ah] @ SoH 100
    capa_min = specs.capa["min"]  # [Ah] @ SoH 100
    capa_soc_crit_max = specs.capa["soc_crit_max"]
    capa_soc_crit_min = specs.capa["soc_crit_min"]
    capa_soc_max = specs.capa["soc_max"]
    capa_soc_min = specs.capa["soc_min"]
    soc_start = capa_soc_min

    curr_profiles = np.load("doc/curr_plot.npy", allow_pickle=True)[:, 20_000:30_000]

    # Time axis
    time = np.arange(curr_profiles.shape[1])
    # Create the plot with size suitable for a GitHub README

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"hspace": 0}
    )

    # Highlight the background areas for current plot with solid colors

    ax1.fill_between(
        time,
        specs.I_terminal["soc_crit_dchg"],
        specs.I_terminal["soc_crit_chg"],
        color="lightgreen",
        alpha=0.5,
    )

    ax1.fill_between(
        time,
        specs.I_terminal["dchg"],
        specs.I_terminal["soc_crit_dchg"],
        color="orange",
        alpha=0.5,
    )

    ax1.fill_between(
        time,
        specs.I_terminal["short_dchg"],
        specs.I_terminal["dchg"],
        color="lightcoral",
        alpha=0.5,
    )

    ax1.fill_between(
        time,
        specs.I_terminal["soc_crit_chg"],
        specs.I_terminal["short_chg"],
        color="lightcoral",
        alpha=0.5,
    )

    # Plot the current profiles

    ax1.plot(time, curr_profiles[1, :], label="Profile 1", color="blue")

    ax1.plot(time, curr_profiles[0, :], label="Profile 2", color="brown")

    ax1.plot(time, curr_profiles[2, :], label="Profile 3", color="black")

    ax1.axhline(y=specs.I_terminal["chg"], color="gray", linestyle=":", label="I_chg")

    ax1.axhline(y=specs.I_terminal["dchg"], color="gray", linestyle=":", label="I_dchg")

    ax1.axhline(
        y=specs.I_terminal["short_chg"],
        color="gray",
        linestyle=":",
        label="I_short_chg",
    )

    ax1.axhline(
        y=specs.I_terminal["short_dchg"],
        color="gray",
        linestyle=":",
        label="I_short_dchg",
    )

    ax1.axhline(
        y=specs.I_terminal["soc_crit_chg"],
        color="gray",
        linestyle=":",
        label="I_soc_crit_chg",
    )

    ax1.axhline(
        y=specs.I_terminal["soc_crit_dchg"],
        color="gray",
        linestyle=":",
        label="I_soc_crit_dchg",
    )

    # Set y-axis labels for current plot

    ax1.set_yticks(
        [
            specs.I_terminal["chg"],
            specs.I_terminal["dchg"],
            specs.I_terminal["short_chg"],
            specs.I_terminal["short_dchg"],
            specs.I_terminal["soc_crit_chg"],
            specs.I_terminal["soc_crit_dchg"],
        ]
    )

    ax1.set_yticklabels(
        [
            f'{specs.I_terminal["chg"]:.2f}',
            f'{specs.I_terminal["dchg"]:.2f}',
            f'{specs.I_terminal["short_chg"]:.2f}',
            f'{specs.I_terminal["short_dchg"]:.2f}',
            f'{specs.I_terminal["soc_crit_chg"]:.2f}',
            f'{specs.I_terminal["soc_crit_dchg"]:.2f}',
        ]
    )

    ax1.set_ylabel("Terminal Current [A]")

    ax1.grid(False)

    # Remove the bottom spine of the top plot

    ax1.spines["bottom"].set_visible(False)

    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # Generate a mock SOC profile for demonstration (replace this with actual SOC calculation if available)

    soc = (
        np.abs(np.cumsum(curr_profiles, axis=1) / 3600 / specs.capa["max"])
        + specs.capa["soc_min"]
    )

    # Plot the SOC profiles

    ax2.plot(time, soc[1, :], label="Profile 1", color="blue")

    ax2.plot(time, soc[0, :], label="Profile 2", color="brown")

    ax2.plot(time, soc[2, :], label="Profile 3", color="black")

    ax2.axhline(
        y=specs.capa["soc_crit_max"], color="gray", linestyle=":", label="SOC Crit Max"
    )

    ax2.axhline(
        y=specs.capa["soc_crit_min"], color="gray", linestyle=":", label="SOC Crit Min"
    )

    ax2.axhline(y=specs.capa["soc_max"], color="gray", linestyle=":", label="SOC Max")

    ax2.axhline(y=specs.capa["soc_min"], color="gray", linestyle=":", label="SOC Min")

    # Highlight the background area between the lowest two and the upper two SOC labels in light green

    ax2.fill_between(
        time,
        specs.capa["soc_crit_min"],
        specs.capa["soc_crit_max"],
        color="lightgreen",
        alpha=0.5,
    )

    # Highlight the areas below SOC Crit Min and above SOC Crit Max in light red

    ax2.fill_between(
        time,
        specs.capa["soc_min"],
        specs.capa["soc_crit_min"],
        color="lightcoral",
        alpha=0.5,
    )

    ax2.fill_between(
        time,
        specs.capa["soc_crit_max"],
        specs.capa["soc_max"],
        color="lightcoral",
        alpha=0.5,
    )

    # Set y-axis labels for SOC plot

    ax2.set_yticks(
        [
            specs.capa["soc_crit_max"],
            specs.capa["soc_crit_min"],
            specs.capa["soc_min"],
            specs.capa["soc_max"],
        ]
    )

    ax2.set_yticklabels(
        [
            f'{specs.capa["soc_crit_max"]:.2f}',
            f'{specs.capa["soc_crit_min"]:.2f}',
            f'{specs.capa["soc_min"]:.2f}',
            f'{specs.capa["soc_max"]:.2f}',
        ]
    )

    ax2.set_xlabel("Sequence Time [s]")

    ax2.set_ylabel("SoC")

    ax2.grid(False)

    # Remove the top spine of the bottom plot

    ax2.spines["top"].set_visible(False)

    # Place the legend in the center below the plot

    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.05))

    # Ensure the x-axis starts at 0 and ends at the last data point

    ax2.set_xlim(left=0, right=time[-1])

    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)
    plt.show()
    tt = 5


################################ Dataset snippet ################################
# file_path = "data/train/dummy_battery_data.h5"
# dataset_name = "random"
# with h5py.File(file_path, "r+") as file:
#     dataset = file[dataset_name]
#     verify_dataset_limits(dataset)
#     # scale_data(file_path, dataset_name)
#     x = 5
#     # plt.plot(file["random"][0,0,0])
#     fig, ax1 = plt.subplots()
#
#     color = "tab:red"
#     ax1.set_xlabel("Time")
#     ax1.set_ylabel("Random", color=color)
#     ax1.plot(file["random"][0, :, 0], color=color)
#     ax1.tick_params(axis="y", labelcolor=color)
#
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     color = "tab:blue"
#     ax2.set_ylabel("Random Scaled", color=color)
#     ax2.plot(file["random_scaled"][0, :, 0], "--", color=color)
#     ax2.tick_params(axis="y", labelcolor=color)
#
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.show()
