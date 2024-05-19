import os

import matplotlib.pyplot as plt
import numpy as np

from config_data import BatteryDatasheet
from drive_cycle_bin import load_bins_from_csv


def normalize_to_range(data):
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    if range_val <= 0:
        data[:] = 1
        return data
    else:
        # Normalize such that the current values are in the range [0, 1]
        data = (data - min_val) / range_val
        return data


def get_field_current_1(soc_tgt, capa, steps, dt, pos_bin, neg_bin):
    def generate_current(bin_data, length):
        curr = []
        while len(curr) <= length:
            n = np.random.default_rng().integers(0, len(bin_data))
            curr.extend(bin_data[n][:, 1])
        return np.array(curr[:length])

    if steps == 0:
        curr = 0
        return curr

    if soc_tgt > 0:
        curr = generate_current(pos_bin, steps)
        curr_max = soc_tgt * capa / steps / dt  # [A]
        curr = normalize_to_range(np.abs(curr)) * curr_max
    elif soc_tgt < 0:
        curr = generate_current(neg_bin, steps)
        curr_max = soc_tgt * capa / steps / dt  # [A]
        curr = normalize_to_range(np.abs(curr)) * curr_max
    else:
        curr = np.zeros(steps)

    return curr


def get_field_current(soc_tgt, capa, steps, dt, pos_bin, neg_bin):
    if soc_tgt > 0:
        curr = []
        while len(curr) <= len(steps):
            n = np.random.default_rng().integers(0, len(pos_bin))
            curr.extend(pos_bin[n])
        curr_max = soc_tgt * capa / len(steps) / dt  # [A]
        # truncate to steps and scale to max possible current
        curr = np.abs(np.array(curr))
        curr = curr[: len(steps)] * curr_max
    if soc_tgt < 0:
        curr = []
        while len(curr) <= len(steps):
            n = np.random.default_rng().integers(0, len(neg_bin))
            curr.extend(neg_bin[n])
        curr_max = soc_tgt * capa / len(steps) / dt  # [A]
        # truncate to steps and scale to max possible current
        curr = np.abs(np.array(curr))
        curr = curr[: len(steps)] * curr_max
    else:
        curr = np.zeros(len(steps))

    return curr


def static_current(specs):
    """
    Generating a static like profile type using a step function
    with the following constraints:
    all params sampled randomly and uniformly
    specified constant sequence time (seq_len)
    >= 1 full cycle (charging, dischargin)
    I_max and I_min
    min and max SoC (for stable simulation)
    special current treatment for 3<SoC<20 & 80<SoC<97
    after every step follows a 0 current hold (equillibrium information)
    every consecutive step has a different sign (polarization effects)
    first full charging (with small discharge pulses, see above) after discharging (with small charging pulses)
    """
    capa = specs.capa["max"] * 3600  # [As]
    soc_min = specs.capa["soc_min"]
    soc_max = specs.capa["soc_max"]
    dt = specs.dt  # step size
    curr_crit_chg = specs.I_terminal["soc_crit_chg"]
    curr_cont_chg = specs.I_terminal["chg"]
    curr_short_chg = specs.I_terminal["short_chg"]
    curr_crit_dchg = specs.I_terminal["soc_crit_dchg"]
    curr_cont_dchg = specs.I_terminal["dchg"]
    curr_short_dchg = specs.I_terminal["short_dchg"]
    soc_crit_min = specs.capa["soc_crit_min"]
    soc_crit_max = specs.capa["soc_crit_max"]
    time_short = specs.I_terminal["short_time"]

    curr_min = 3.2  # [A]
    last_hold = 30 / dt  # last 0 duration [steps]
    sequence = np.zeros(specs.seq_len)  # [steps]
    seq_len = specs.seq_len - last_hold
    mode = "chg"  # chg, dchg
    soc = specs.capa["soc_min"]
    idx = 8  # start with 0 current for 4 steps (initial equilibrium)
    pos_file = os.path.join("..", "data", "current", "drive_cycle", "pos_bins.csv")
    neg_file = os.path.join("..", "data", "current", "drive_cycle", "neg_bins.csv")
    pos_bins, neg_bins = load_bins_from_csv(pos_file, neg_file)
    soc_tgt = 1

    params = [
        [curr_crit_chg, curr_short_chg, curr_cont_chg],
        [curr_crit_dchg, curr_short_dchg, curr_cont_dchg],
    ]
    while idx < seq_len:
        for i, param in enumerate(params):
            soc_tgt = sample_soc_tgt(mode, soc, soc_min, soc_max, abs(soc_tgt))[
                i
            ]  # (+pos) [.], (-neg) [.]
            steps = sample_duration(
                *param,
                soc,
                soc_tgt,
                soc_crit_min,
                soc_crit_max,
                capa,
                curr_min,
                time_short,
                dt,
                seq_len,
                idx,
            )  # int: n_steps based on dt

            # curr = soc_tgt * capa / duration  # [A]
            curr = get_field_current_1(
                soc_tgt, capa, steps, dt, pos_bins, neg_bins
            )  # [A]

            # Apply charging current and hold step
            sequence[idx : idx + steps] = curr
            idx += steps

            if steps <= 1:
                soc_tgt = curr * steps * dt / capa
                soc += soc_tgt
            else:
                soc_tgt = np.trapz(curr, dx=dt) / capa
                soc += soc_tgt

            duration_hold = np.random.uniform(15, 30)
            steps_hold = int(min(duration_hold / dt, seq_len - idx))
            sequence[idx : idx + steps_hold] = 0
            idx += steps_hold

            if idx >= seq_len:
                break

        if soc >= 0.99 * soc_max:
            mode = "dchg"
        elif soc <= 1.01 * soc_min:
            mode = "chg"

    return sequence


def sample_soc_tgt(mode, soc, soc_min, soc_max, soc_swap=1):
    # Option to restrict discharge/charge in crit soc area to enforce full chg/dchg
    if mode == "chg":
        high = soc_max - soc
        soc_chg = np.random.default_rng().uniform(0, high)
        soc_dchg = np.random.default_rng().uniform(0, min(soc_swap, soc - soc_min))
        return soc_chg, -soc_dchg

    else:
        high = soc - soc_min
        soc_dchg = np.random.default_rng().uniform(0, high)
        soc_chg = np.random.default_rng().uniform(0, min(soc_swap, soc_max - soc))
        return -soc_dchg, soc_chg


def sample_duration(
    curr_crit,
    curr_short,
    curr_cont,
    soc,
    soc_tgt,
    soc_crit_min,
    soc_crit_max,
    capa,
    curr_min,
    time_short,
    dt,
    seq_len,
    idx,
):
    if soc < soc_crit_min or soc > soc_crit_max:
        duration_min = soc_tgt * capa / curr_crit

    elif soc + soc_tgt > soc_crit_max or soc + soc_tgt < soc_crit_min:
        duration_min = soc_tgt * capa / curr_crit

    elif time_short * curr_short >= soc_tgt:
        duration_min = soc_tgt * capa / curr_short

    else:
        duration_min = soc_tgt * capa / curr_cont

    duration_min = abs(duration_min)
    duration_max = abs(soc_tgt * capa / curr_min)  # [s]
    alpha, beta = 1, 4  # This will skew the distribution towards the lower bound
    scale = duration_max - duration_min
    location = duration_min
    duration = location + np.random.default_rng().beta(alpha, beta) * scale
    duration = int(np.ceil(duration))
    steps = int(min(duration / dt, seq_len - idx))
    return steps


def field_current(specs):
    """
    Generating a real world like profile type using drive cycles
    with the following constraints:
    all params sampled randomly and uniformly from drive cycle buckets (charger, discharge)
    specified constant sequence time (seq_len)
    >= 1 full cycle (charging, dischargin)
    I_max and I_min
    min and max SoC (for stable simulation)
    special current treatment for 3<SoC<20 & 80<SoC<97
    after every single sample a 0 oreder current hold follows
    every consecutive sampling has a different sign and also follows a shorter 0 order hold (polarization effects)
    first full charging (with small discharge pulses, see above) after discharging (with small charging pulses)
    """
    pass


def dynamic_current(specs):
    """
    Generating a dynamic like profile type superimposing static and fiel pofile and applying a smothing kernel
    with the following constraints:
    all params sampled randomly and uniformly
    specified constant sequence time (seq_len)
    >= 1 full cycle (charging, dischargin)
    I_max and I_min
    min and max SoC (for stable simulation)
    special current treatment for 3<SoC<20 & 80<SoC<97
    deliberatly no zero order hold, fully dynamic
    every consecutive step has a different sign (polarization effects)
    first full charging (with small discharge pulses, see above) after discharging (with small charging pulses)
    """
    # call static, call field, add both, smooth superimposed profile
    # moving scaling window, window moves block wise for time_short
    # for every move a new scaling to stay within boundaries and ensure chg/dchg
    pass


def generate_current_profiles():
    """
    Sampling uniformly over the current limits is curcial to get a percise model
    for battery cell bahaviour. If we would to model a real world scenario, like
    power/energy utilization we would sampel form a normal like distribution. But
    here we are interested in the battery bahaviour over the full range of the
    specifications, therefore a balanced omount of any current amplitude at any soc for
    any duration!
    """
    np.random.seed(420)
    profiles = [static_current]  # , dynamic_current, field_current]
    n_profiles = 150  # Total of: n_profiles * profiles.len()
    specs = BatteryDatasheet()

    output = []
    for _ in range(n_profiles):
        output.extend([profile(specs) for profile in profiles])
    # negative sign for charging!
    output = -np.array(output)
    return output

    # test output for specs


if __name__ == "__main__":
    generate_current_profiles()
