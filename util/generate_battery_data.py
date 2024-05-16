import matplotlib.pyplot as plt
import numpy as np

from config_data import BatteryDatasheet


def static_current_5(specs):
    """
    Generate a profile sequence using dynamic bounds based on current SoC and remaining time.
    """
    sequence = np.zeros(specs.seq_len)
    soc = specs.capa["soc_min"]
    capa = specs.capa["max"]
    idx = 4  # start with 0 current for 4 steps (initial equilibrium)
    mode = "chg"
    last_hold = 30 / specs.dt
    test_1, test_2, test_3, test_4, test_5, = [], [], [], [], []

    while idx < specs.seq_len:
        #FIX: skewed distribution change when mode changes and put in function mode == chg
        # Current sampling: 0 to max charge current, adjust for critical SoC range
        if soc < specs.capa["soc_crit_min"] or soc > specs.capa["soc_crit_max"]:
            chg_current = np.random.uniform(specs.I_terminal["soc_crit_chg"], 0)
            # chg_current = chg_current_crit.uniform(specs.I_terminal["soc_crit_chg"], 0)
            test_1.append(chg_current)
        else:
            chg_current = np.random.uniform(specs.I_terminal["short_chg"], 0)
            # chg_current = chg_current_short.uniform(specs.I_terminal["short_chg"], 0)
            test_2.append(chg_current)

        # max_possible to ensure dynamic sampling bounds based on current soc
        max_possible = -(specs.capa["soc_max"] - soc) * capa * 3600 / specs.dt
        current = max(max_possible, chg_current)
        test_3.append(chg_current)

        # Duration sampling with a dynamic upper bound to avoid violating constraints
        if abs(current) > abs(specs.I_terminal["chg"]):
            # Check remaining time to soc_crit max
            soc_to_crit = soc + abs(current) * specs.I_terminal["short_time"]/specs.dt/3600 / capa
            upper_bound = (
                specs.I_terminal["short_time"]
                if soc_to_crit < specs.capa["soc_crit_max"]
                else (specs.capa["soc_crit_max"] - soc) * capa * 3600 / abs(current)
            )
            duration = np.random.uniform(specs.dt, upper_bound)
            test_4.append(duration)
        else:
            # Ensure time left for at least 1 min of steps
            remaining_steps = int(
                ((specs.capa["soc_max"] - soc) * capa * 3600 / abs(current) / specs.dt)
                - last_hold
            )
            duration = np.random.uniform(specs.dt, remaining_steps)
            test_5.append(duration)

        # current_steps = int(min(duration, specs.seq_len - idx))
        current_steps = int(min((1 - soc)*duration, specs.seq_len - idx))

        # Apply charging current and hold step
        sequence[idx : idx + current_steps] = current
        idx += current_steps

        # Update SoC based on the current and duration applied
        soc += (abs(current) * current_steps * specs.dt) / (capa * 3600)

        # Hold step duration sampling
        hold_duration = np.random.uniform(5, 15)
        hold_steps = int(min(hold_duration / specs.dt, specs.seq_len - idx))
        sequence[idx : idx + hold_steps] = 0
        idx += hold_steps

        #FIX: skewed distribution change when mode changes and put in function mode == chg
        # Short discharge pulse for polarization effect
        if soc > specs.capa["soc_crit_max"] or soc < specs.capa["soc_crit_min"]:
            dchg_current = np.random.uniform(0, specs.I_terminal["soc_crit_dchg"])
        else:
            dchg_current = np.random.uniform(0, specs.I_terminal["short_dchg"])

        max_possible = (soc - specs.capa["soc_min"]) * capa * 3600 / specs.dt
        current = min(max_possible, dchg_current)

        if abs(current) > abs(specs.I_terminal["dchg"]):
            # Check remaining time to soc_crit min
            soc_to_crit = (soc - abs(current) * specs.I_terminal["short_time"] /specs.dt/3600/capa)
            upper_bound = (
                specs.I_terminal["short_time"]
                if soc_to_crit > specs.capa["soc_crit_min"]
                else (soc - specs.capa["soc_crit_min"]) * capa * 3600 / abs(current)
            )
            duration = np.random.uniform(specs.dt, upper_bound)
        else:
            # Ensure time left for at least 30 sec of steps
            remaining_steps = int(((specs.seq_len - idx) / specs.dt) - last_hold)
            # upper_max = np.random.uniform(swap_dchg*soc**2, swap_dchg) * capa * 3600 / specs.dt
            upper_bound = min(
                remaining_steps,
                # upper_max,
                (soc - specs.capa["soc_min"])
                * capa
                * 3600
                / abs(dchg_current)
                / specs.dt,
            )
            duration = np.random.uniform(specs.dt, upper_bound)

        discharge_steps = int(min(soc * duration, specs.seq_len - idx))

        sequence[idx : idx + discharge_steps] = current
        idx += discharge_steps

        # Update SoC after discharge pulse
        soc -= (current * discharge_steps * specs.dt) / (capa * 3600)

        # Hold step duration after discharge
        if specs.seq_len - idx <= last_hold:
            hold_steps2 = int(specs.seq_len - idx)
        else:
            hold_duration2 = np.random.uniform(5, 15)
            hold_steps2 = int(min(hold_duration2 / specs.dt, specs.seq_len - idx))
        sequence[idx : idx + hold_steps2] = 0
        idx += hold_steps2

    return sequence



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
    specs = BatteryDatasheet()

    # get the full seq_len and cycle_time from specs
    # first part of the sequence starts always with 0 for 3 time_setps=specs.dt
    # now we start to create the charging profile (negative current), start capa/soc=specs.capa[soc_min]
    # check if we are in the specs.capa[soc_crit_max] or specs.capa[soc_crit_min] range
    # I: sample from numpy uniform a current amplitude and a step_duration (time):

    # - current: min, max based on where we are: max, min in general
    #    if in specific region like short or soc_crit the according values

    # - time: min=dt, max=...not exeding capa, we need to track capa/soc always, also not exeeding
    #    short_time if the current value is more than general max or min, also not violating a
    #    transition in a region like crit, where the amplitude would violate the constraints

    # - II: after the pulse we follow with a 0 amplitude for a sampled time -equillibrium)
    # what should be suffieciently large to let the cell satlle (maybe sth like 5-15 sec)

    # - III: the next step has to have a positive sign for discharging (polarization)
    # with a reasonable amplitude and also duration (short(ish), we want to overall achieve
    # charging, but have the plarization effect)

    # - now again 0 amplitude for a short time

    # - now charging pulse again, essentally repeat (I - III) until the running
    # capa/soc == specs.capa["soc_max"]

    # - now to the same for discharge untile running soc==specs.capa["soc_min"]

    # - also the profile should end with II, a 0 amplitude section


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
    profiles = [static_current_5]  # , dynamic_current, field_current]
    n_profiles = 10  # Total of: n_profiles * profiles.len()
    specs = BatteryDatasheet()

    output = []
    for _ in range(n_profiles):
        output.extend([profile(specs) for profile in profiles])
    output = np.array(output)
    return output

    # test output for specs


if __name__ == "__main__":
    generate_current_profiles()
