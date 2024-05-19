# Drive cycles from:
# https://github.com/pybamm-team/PyBaMM/tree/
# 0237b111048fcd451933e121652c426a6280fe55/pybamm/input/drive_cycles
import csv
import os

import numpy as np


def normalize_to_range(data):
    min_val = np.min(data[:, 1])
    max_val = np.max(data[:, 1])
    range_val = max_val - min_val
    # Normalize the current values to the range [-1, 1]
    data[:, 1] = 2 * (data[:, 1] - min_val) / range_val - 1
    return data

def load_bins_from_csv(pos_file, neg_file):
    out = []
    for file in [pos_file, neg_file]:
        bins = {}
        data = np.loadtxt(file, delimiter=",", skiprows=1)
        for row in data:
            bin_index = int(row[2])
            if bin_index not in bins:
                bins[bin_index] = []
            bins[bin_index].append(row[:2])
        out.append([np.array(bins[key]) for key in sorted(bins.keys())])
    pos_bins, neg_bins = out[0], out[1]
    return pos_bins, neg_bins


def save_bins_to_csv(pos_bins, neg_bins, pos_file, neg_file):
    with open(pos_file, "w", newline="") as pos_csvfile:
        pos_writer = csv.writer(pos_csvfile, delimiter=",")
        pos_writer.writerow(["time", "current", "bin_index"])  # Write header
        for bin_index, bin in enumerate(pos_bins):
            for row in bin:
                pos_writer.writerow([row[0], row[1], bin_index])

    with open(neg_file, "w", newline="") as neg_csvfile:
        neg_writer = csv.writer(neg_csvfile, delimiter=",")
        neg_writer.writerow(["time", "current", "bin_index"])  # Write header
        for bin_index, bin in enumerate(neg_bins):
            for row in bin:
                neg_writer.writerow([row[0], row[1], bin_index])


def interpolate_to_base_time(data, dt=1):
    base_time = np.arange(0, int(data[-1, 0]) + 1, dt)
    interpolated_currents = np.interp(base_time, data[:, 0], data[:, 1])
    interpolated_data = np.column_stack((base_time, interpolated_currents))
    return interpolated_data


def create_masks_and_split(data):
    sign = np.sign(data[:, 1])
    sign[sign == 0] = 1  # Treat zeros as positive for both bins

    changes = np.where(np.diff(sign) != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(sign)]))

    pos_bins = [
        data[boundaries[i] : boundaries[i + 1]]
        for i in range(len(boundaries) - 1)
        if sign[boundaries[i]] >= 0
    ]
    neg_bins = [
        data[boundaries[i] : boundaries[i + 1]]
        for i in range(len(boundaries) - 1)
        if sign[boundaries[i]] <= 0
    ]

    return pos_bins, neg_bins


def split_drive_cycles(bat_voltage=800, cells_parallel=1):
    us06_path = os.path.join("..", "data", "current", "drive_cycle", "US06.csv")
    udds_path = os.path.join("..", "data", "current", "drive_cycle", "UDDS.csv")
    wltc_path = os.path.join("..", "data", "current", "drive_cycle", "WLTC.csv")

    us06 = np.loadtxt(us06_path, comments="#", delimiter=",")
    udds = np.loadtxt(udds_path, comments="#", delimiter=",")
    wltc = (
        np.loadtxt(wltc_path, comments="#", delimiter=",")
        * 1000
        / bat_voltage
        / cells_parallel
    )
    us06 = interpolate_to_base_time(us06)
    udds = interpolate_to_base_time(udds)
    wltc = interpolate_to_base_time(wltc)

    # us06 = normalize_to_range(us06)
    # udds = normalize_to_range(udds)
    # wltc = normalize_to_range(wltc)

    us06_pos_bins, us06_neg_bins = create_masks_and_split(us06)
    udds_pos_bins, udds_neg_bins = create_masks_and_split(udds)
    wltc_pos_bins, wltc_neg_bins = create_masks_and_split(wltc)

    pos_bins = us06_pos_bins + udds_pos_bins + wltc_pos_bins
    neg_bins = us06_neg_bins + udds_neg_bins + wltc_neg_bins

    save_bins_to_csv(
        pos_bins,
        neg_bins,
        os.path.join("..", "data", "current", "drive_cycle", "pos_bins.csv"),
        os.path.join("..", "data", "current", "drive_cycle", "neg_bins.csv"),
    )

    return pos_bins, neg_bins


if __name__ == "__main__":
    # Example usage and plotting
    pos_bins, neg_bins = split_drive_cycles()
    # Example usage
    pos_file = os.path.join("..", "data", "current", "drive_cycle", "pos_bins.csv")
    neg_file = os.path.join("..", "data", "current", "drive_cycle", "neg_bins.csv")
    pos_bins, neg_bins = load_bins_from_csv(pos_file, neg_file)
