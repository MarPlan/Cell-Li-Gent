import matplotlib.pyplot as plt
import numpy as np
import pybamm


def plot_profiles_3d(inputs, save=False):
    current = inputs[:, :, 0]
    voltage = inputs[:, :, 1]
    temperature = inputs[:, :, 2]
    soc = inputs[:, :, 3]

    # Meshgrid for 3D plot
    grid_x, grid_y = np.mgrid[
        current.min() : current.max() : 50j, voltage.min() : voltage.max() : 50j
    ]
    grid_z = griddata(
        (current.ravel(), voltage.ravel()),
        temperature.ravel(),
        (grid_x, grid_y),
        method="nearest",
        rescale=True,
    )

    # Gaussian filter
    sigma = 1
    grid_z_smth = gaussian_filter(grid_z, sigma=sigma)

    # Adjusted KDE bandwidth
    bandwidth_factor = 3
    kde_3d = gaussian_kde(
        np.vstack([current.ravel(), voltage.ravel(), temperature.ravel()]),
        bw_method=bandwidth_factor,
    )
    density_3d = kde_3d(
        np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z_smth.ravel()])
    ).reshape(grid_x.shape)
    density_3d = np.log(density_3d + 1e-5)

    # 2D KDE densities with adjusted bandwidth
    kde_2d_cv = gaussian_kde(
        np.vstack([current.ravel(), voltage.ravel()]), bw_method=bandwidth_factor
    )
    density_2d_cv = kde_2d_cv(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(
        grid_x.shape
    )
    density_2d_cv = np.log(density_2d_cv + 1e-5)

    grid_x_ct, grid_z_ct = np.mgrid[
        current.min() : current.max() : 50j,
        temperature.min() : temperature.max() : 50j,
    ]
    kde_2d_ct = gaussian_kde(
        np.vstack([current.ravel(), temperature.ravel()]), bw_method=bandwidth_factor
    )
    density_2d_ct = kde_2d_ct(
        np.vstack([grid_x_ct.ravel(), grid_z_ct.ravel()])
    ).reshape(grid_x_ct.shape)
    density_2d_ct = np.log(density_2d_ct + 1e-5)

    grid_y_vt, grid_z_vt = np.mgrid[
        voltage.min() : voltage.max() : 50j,
        temperature.min() : temperature.max() : 50j,
    ]
    kde_2d_vt = gaussian_kde(
        np.vstack([voltage.ravel(), temperature.ravel()]), bw_method=bandwidth_factor
    )
    density_2d_vt = kde_2d_vt(
        np.vstack([grid_y_vt.ravel(), grid_z_vt.ravel()])
    ).reshape(grid_y_vt.shape)
    density_2d_vt = np.log(density_2d_vt + 1e-5)

    # Adjusting the color normalization using the densities
    viridis_inv = mpl.colormaps["viridis"].reversed()
    YlOrRd = mpl.colormaps["YlOrRd"]

    # face_colors_3d = viridis_inv(
    #     (density_3d - density_3d.min()) / (density_3d.max() - density_3d.min())
    # )

    # np.log(density_3d + 1e-5)
    # norm_soc = Normalize(vmin=soc.min(), vmax=soc.max())
    # grid_z_smth = np.log(grid_z_smth + 1e-5)
    # face_colors_3d = viridis_inv((grid_z_smth - grid_z_smth.min()) / (grid_z_smth.max() - grid_z_smth.min()))
    grid_soc = griddata(
        (current.ravel(), voltage.ravel()),
        soc.ravel(),
        (grid_x, grid_y),
        method="nearest",
        rescale=True,
    )
    grid_soc_smth = gaussian_filter(grid_soc, sigma=sigma)
    # grid_soc_smth =  np.log(grid_soc_smth + 1e-5)
    face_colors_3d = viridis_inv(
        (grid_soc_smth - grid_soc_smth.min())
        / (grid_soc_smth.max() - grid_soc_smth.min())
    )

    # face_colors_3d = viridis_inv(norm_soc(grid_z_smth))

    face_colors_2d_cv = YlOrRd(
        (density_2d_cv - density_2d_cv.min())
        / (density_2d_cv.max() - density_2d_cv.min())
    )
    face_colors_2d_ct = YlOrRd(
        (density_2d_ct - density_2d_ct.min())
        / (density_2d_ct.max() - density_2d_ct.min())
    )
    face_colors_2d_vt = YlOrRd(
        (density_2d_vt - density_2d_vt.min())
        / (density_2d_vt.max() - density_2d_vt.min())
    )

    # fig_size_1 = (8, 6)
    fig_size_1 = (5.5, 4)
    # fig.set_size_inches(fig_size_big)
    # Visualization
    fig = plt.figure(figsize=fig_size_1)
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    # 3D surface plot
    surf_3d = ax.plot_surface(
        grid_x,
        grid_y,
        grid_z_smth,
        facecolors=face_colors_3d,
        edgecolor="none",
        alpha=0.9,
    )

    # 2D surface plots
    surf_2d_cv = ax.plot_surface(
        grid_x,
        grid_y,
        np.full_like(grid_z, grid_z.min()),
        facecolors=face_colors_2d_cv,
        edgecolor="none",
        alpha=0.8,
    )
    surf_2d_ct = ax.plot_surface(
        grid_x_ct,
        np.full_like(grid_x_ct, grid_y.max()),
        grid_z_ct,
        facecolors=face_colors_2d_ct,
        edgecolor="none",
        alpha=0.8,
    )
    surf_2d_vt = ax.plot_surface(
        np.full_like(grid_y_vt, grid_x.min()),
        grid_y_vt,
        grid_z_vt,
        facecolors=face_colors_2d_vt,
        edgecolor="none",
        alpha=0.8,
    )

    ax.set_xlabel("Terminal Current [A]")
    ax.set_ylabel("Terminal Voltage [V]")
    ax.set_zlabel("Surface Temperature [°C]")
    # ax.set_xlim([current.min(), current.max()])
    # ax.set_ylim([voltage.min(), voltage.max()])
    # ax.set_zlim([temperature.min(), temperature.max()])
    ax.set_xlim([grid_x_ct.min(), grid_x_ct.max()])
    ax.set_ylim([grid_y_vt.min(), grid_y_vt.max()])
    ax.set_zlim([grid_z.min(), grid_z.max()])
    # Normalize# Create a new axis for the colorbar at the desired position below the main plot
    cbar_ax_2d = fig.add_axes([0.3, +0.12, 0.5, 0.03])  # [left, bottom, width, height]
    cbar_ax_3d = fig.add_axes([0.3, +0.08, 0.5, 0.03])  # [left, bottom, width, height]

    norm = Normalize(vmin=0, vmax=1)
    # Add colorbars for both colormaps
    sm = plt.cm.ScalarMappable(cmap=mpl.colormaps["viridis"], norm=norm)
    cbar_3d = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.5,
        aspect=15,
        pad=-0.1,
        cax=cbar_ax_3d,
        orientation="horizontal",
    )
    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=YlOrRd, norm=norm)
    cbar_2d = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.5,
        aspect=15,
        pad=0.15,
        cax=cbar_ax_2d,
        orientation="horizontal",
    )

    cbar_3d.set_label("Density/SoC", labelpad=-10)
    cbar_3d.set_ticks([0, 1])
    cbar_3d.set_ticklabels(["Low", "High"])

    cbar_2d.set_ticks([])

    # t = Bbox([[0.5, -0.5], [6, 3.5]])
    fig.tight_layout()
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    from matplotlib.transforms import Bbox
    t = Bbox([[0.5, 0], [6, 4]])
    if save:
        # plt.savefig("doc/data_3d.png", dpi=300)
        plt.savefig( "doc/data_3d.pdf", format="pdf", bbox_inches=t, dpi=300)

    plt.show()


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(".."))
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pybamm
    from matplotlib.colors import Normalize
    from model.battery import Lumped, LumpedSurface
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde

    os.chdir("..")
    capa = 5
    outputs = [
        "Current [A]",
        "Terminal voltage [V]",
        "Volume-averaged surface temperature [C]",
        "Discharge capacity [A.h]",
        # "Battery open-circuit voltage [V]"
    ]
    profile_0 = pybamm.load("data/profile_0.pkl")
    profile_1 = pybamm.load("data/profile_1.pkl")
    profile_2 = pybamm.load("data/profile_2.pkl")

    profile_0 = {
        key: (
            profile_0[key].entries / capa
            if key == outputs[-1]
            else profile_0[key].entries
        )
        for key in outputs
    }
    profile_1 = {
        key: (
            profile_1[key].entries / capa
            if key == outputs[-1]
            else profile_1[key].entries
        )
        for key in outputs
    }
    profile_2 = {
        key: (
            profile_2[key].entries / capa
            if key == outputs[-1]
            else profile_2[key].entries
        )
        for key in outputs
    }
    profiles = [profile_0, profile_1, profile_2]

    # Combine profiles into a single input array with shape (x, y, 4) where the last dimension is [Current, Voltage, Temperature, SoC]
    input_data = np.stack(
        [
            np.column_stack(
                [
                    prof["Current [A]"],
                    prof["Terminal voltage [V]"],
                    prof["Volume-averaged surface temperature [C]"],
                    prof["Discharge capacity [A.h]"],
                ]
            )
            for prof in profiles
        ],
        axis=1,
    )
    plot_profiles_3d(input_data, save=True)

    # Define slice range for the second column
    slice_range = slice(44_500, 47_500)
    # Create a 4x2 plot
    fig, axs = plt.subplots(4, 2, figsize=(10, 14), sharex="col", sharey="row")
    # Define y-axis labels
    y_labels = [
        "SoC [·]",
        "Surface Temperature [°C]",
        "Terminal Voltage [V]",
        "Terminal Current [A]",
    ]
    # Define colors and linestyles for the profiles
    colors = ["b", "g", "r"]
    linestyles = ["-", "-", "-"]
    # Plot the profiles in the first column and the sliced profiles in the second column
    time = np.linspace(0, profile_0[outputs[0]].size / 3600, profile_0[outputs[0]].size)
    for row, key in enumerate(reversed(outputs)):  # reversed to match bottom to top
        # Iterate over profiles for the first column
        for idx, prof in enumerate(profiles):
            axs[row, 0].plot(
                time,
                prof[key],
                label=f"Profile {idx}",
                color=colors[idx],
                linestyle=linestyles[idx],
            )
        axs[row, 0].set_ylabel(y_labels[row])
        if row == 3:
            axs[row, 0].set_xlabel("Time [h]")
        if row == 0:
            # axs[row, 0].legend()
            axs[row, 0].set_title("Full Profiles")
        # Iterate over profiles for the second column (sliced range)
        for idx, prof in enumerate(profiles):
            slice_time = time[slice_range]
            # slice_time = np.arange(prof[key].size)[slice_range]
            axs[row, 1].plot(
                slice_time,
                prof[key][slice_range],
                label=f"Profile {idx}",
                color=colors[idx],
                linestyle=linestyles[idx],
            )
        if row == 3:
            axs[row, 1].set_xlabel("Time [h]")
        if row == 0:
            # axs[row, 1].legend()
            axs[row, 1].set_title("Cropped Profiles")
    # Save the figure with appropriate dimensions and dpi
    fig.tight_layout()
    # plt.savefig("doc/profiles_plot.png", dpi=300)
    plt.savefig(
        "doc/profiles_plot.pdf",
        format="pdf",
        bbox_inches="tight",
        # pad_inches=[0, 0, 1, 0]
        # pad_inches="tight"
        dpi=300,
    )

    plt.show()
