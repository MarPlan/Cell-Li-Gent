import matplotlib.pyplot as plt
import numpy as np

from config_data import BatteryDatasheet


def static_current_4(specs):
    """
    Generate a profile sequence using dynamic bounds based on current SoC and remaining time.
    """
    sequence = np.zeros(specs.seq_len)
    soc = specs.capa["soc_min"]
    capa = specs.capa["max"]
    idx = 4  # start with 0 current for 4 steps (initial equilibrium)
    chg_soc = 0.01  # max charge SoC for one sampled duration (upper bound for duration/time)
    dchg_soc = 0.008  # max discharge SoC for one sampled duration (upper bound for duration/time)
    prio = "chg"

    while idx < specs.seq_len:
        # Current sampling: 0 to max charge current, adjust for critical SoC range
        chg_current = np.random.uniform(
            specs.I_terminal["soc_crit_chg"] if soc < specs.capa["soc_crit_min"] or soc > specs.capa["soc_crit_max"]
            else specs.I_terminal["short_chg"],
            0,
        )
        # max_possible to ensure dynamic sampling bounds based on current soc
        max_possible = -(specs.capa["soc_max"] - soc) * capa * 3600 / specs.dt

        current = max(max_possible, chg_current)

        # Duration sampling with a dynamic upper bound to avoid violating constraints
        if abs(current) > abs(specs.I_terminal["chg"]):
            # Check how much time left to soc_crit, else short_time
            upper_bound = specs.I_terminal["short_time"]
        else:
            remaining_steps = int((specs.capa["soc_max"] - soc) * capa * 3600 / abs(current) / specs.dt)
            upper_max = chg_soc * capa * 3600 / abs(current) / specs.dt
            upper_bound = min(remaining_steps, upper_max)

        duration = np.random.uniform(specs.dt, upper_bound)
        current_steps = int(min(duration, specs.seq_len - idx))

        # Apply charging current and hold step
        sequence[idx:idx + current_steps] = current
        idx += current_steps

        # Update SoC based on the current and duration applied
        soc += (abs(current) * current_steps * specs.dt) / (capa * 3600)

        if soc >= specs.capa["soc_max"]*0.99:
            prio = "dchg"

        # Hold step duration sampling
        hold_duration = np.random.uniform(5, 15)
        hold_steps = int(min(hold_duration / specs.dt, specs.seq_len - idx))
        sequence[idx:idx + hold_steps] = 0
        idx += hold_steps

        # Short discharge pulse for polarization effect
        dchg_current = np.random.uniform(
            0,
            specs.I_terminal["soc_crit_dchg"] if soc > specs.capa["soc_crit_max"] or soc < specs.capa["soc_crit_min"]
            else specs.I_terminal["short_dchg"],
        )
        max_possible = (soc - specs.capa["soc_min"]) * capa * 3600 / specs.dt
        current = min(max_possible, dchg_current)

        if abs(current) > abs(specs.I_terminal["dchg"]):
            upper_bound = specs.I_terminal["short_time"]
        else:
            remaining_steps = int((specs.seq_len - idx) / specs.dt)
            upper_max = dchg_soc * capa * 3600 / abs(current) / specs.dt
            upper_bound = min(remaining_steps, upper_max, (soc - specs.capa["soc_min"]) * capa * 3600 / abs(dchg_current) / specs.dt)

        duration = np.random.uniform(specs.dt, upper_bound)
        discharge_steps = int(min(duration, specs.seq_len - idx))

        sequence[idx:idx + discharge_steps] = current
        idx += discharge_steps

        # Update SoC after discharge pulse
        soc -= (current * discharge_steps * specs.dt) / (capa * 3600)

        if soc <= specs.capa["soc_min"]*1.01:
            prio = "chg"

        # Hold step duration after discharge
        hold_duration2 = np.random.uniform(5, 15)
        hold_steps2 = int(min(hold_duration2 / specs.dt, specs.seq_len - idx))
        sequence[idx:idx + hold_steps2] = 0
        idx += hold_steps2

        if prio == "chg":
            chg_soc = 0.01  # max charge SoC for one sampled duration (upper bound for duration/time)
            dchg_soc = 0.008  # max discharge SoC for one sampled duration (upper bound for duration/time)

        if prio == "dchg":
            chg_soc = 0.008  # max charge SoC for one sampled duration (upper bound for duration/time)
            dchg_soc = 0.01  # max discharge SoC for one sampled duration (upper bound for duration/time)

    return sequence


def static_current_3(specs):
    """
    We keep it as random as possible (uniformly). The only thing, beside the specs, we
    have to ensure is that we use the full seq_len and not finish in a fraction of the
    specified time and stay constant/0 for the remaining. Enforce dynamics based on 
    current soc and remaining time!
    """
    sequence = np.zeros(specs.seq_len)
    soc = specs.capa["soc_min"]
    capa = specs.capa["max"]
    idx = 4  # start with 0 current for 4 steps (initial equilibrium)
    chg_soc = 0.01 # max chg_soc for one sampled duration (upperbound for duration/time)
    dchg_soc = 0.01 # max chg_soc for one sampled duration (upperbound for duration/time)

    while idx < specs.seq_len:
        # Current sampling: 0 to max charge current, adjust for critical SoC range
        # negative value for charging
        chg_current = np.random.uniform(
            specs.I_terminal["short_chg"]
            if soc > specs.capa["soc_crit_min"] or soc < specs.capa["soc_crit_max"]
            else specs.I_terminal["soc_crit_chg"],
            0,
        )
        # max_possible to ensure dynamic sampling bounds based on current soc
        max_possible = -(specs.capa["soc_max"]- soc) * capa * 3600 / specs.dt
        # becaus of neg current
        current = max(max_possible, chg_current)

        # Duration sampling with a dynamic upper bound to avoid violating constraints
        if abs(current) > abs(specs.I_terminal["chg"]):
            # FIX: check how much time left to soc_crit, else short_time
            upper_bound = specs.I_terminal["short_time"]
        else:
            # FIX: deduct offset time steps to ensure 0 pulse, discharge, and last 0 pulse for end of sequence (around 1min)
            remaining_steps = int(
                (specs.capa["soc_max"] - soc) * capa * 3600 / abs(current) / specs.dt
            )
            # FIX: upper_max scaling verification
            # max based on specified soc change for one duration
            upper_max = chg_soc*specs.capa["max"]*3600/abs(current)/specs.dt
            upper_bound = ( min(remaining_steps, upper_max)) 

        duration = np.random.uniform(specs.dt, upper_bound)
        current_steps = int(min(duration, specs.seq_len - idx))

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

        # FIX: upper_max scaling verification based on remaining time to enable a full charge but also not to quick, we have to discharge appropriately!!

        # Short discharge pulse for polarization effect
        # Discharge current sampling adjusted for critical SoC range
        dchg_current = np.random.uniform(
            0,
            specs.I_terminal["short_dchg"]
            if soc > specs.capa["soc_crit_min"] or soc < specs.capa["soc_crit_max"]
            else specs.I_terminal["soc_crit_dchg"],
        )
        # max_possible to ensure dynamic sampling bounds based on current soc
        max_possible = (soc-specs.capa["soc_min"]) * capa * 3600 / specs.dt
        # becaus of pos current
        current = min(max_possible, dchg_current)

        # Duration sampling with a dynamic upper bound to avoid violating constraints
        if abs(current) > abs(specs.I_terminal["dchg"]):
            # FIX: check how much time left to soc_crit, else short_time
            upper_bound = specs.I_terminal["short_time"]
        else:
            # FIX: deduct offset time steps to ensure 0 pulse, discharge, and last 0 pulse for end of sequence (around 10sec)
            remaining_steps = int(
                (specs.seq_len - idx) / specs.dt
            )
            # FIX: upper_max scaling verification
            upper_max = dchg_soc*specs.capa["max"]*3600/abs(current)/specs.dt # in steps
            upper_bound = (
                min(
                    remaining_steps,
                    upper_max,
                    (soc-specs.capa["soc_min"]) * capa * 3600 / abs(dchg_current) / specs.dt
                    )
            )  # A fraction for safety margin 1% of the total possible sequence

        duration = np.random.uniform(specs.dt, upper_bound)
        discharge_steps = int(min(duration, specs.seq_len - idx))

        # discharge_duration = np.random.uniform(specs.dt, 15)
        # discharge_steps = int(min(discharge_duration / specs.dt, specs.seq_len - idx))
        sequence[idx : idx + discharge_steps] = current
        idx += discharge_steps

        # Update SoC after discharge pulse
        soc -= (current * discharge_steps * specs.dt) / (capa * 3600)

        # FIX: For last step set explicitly remaining time like duration = remaining time if remaining time <=15sec!
        # Hold step duration after discharge
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
    profiles = [static_current_4]  # , dynamic_current, field_current]
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
