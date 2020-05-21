import numpy as np
from num_experiments.run_model import run_model
from src.utils.sp_utils import *
import pickle
from tqdm.auto import tqdm
from copy import deepcopy
from num_experiments.params_gen import generate_params
import os
from utils.gen_utils import create_dir_if_not_exist, get_project_root


def run_simulations(params, folder_save_to):
    generate_params(1, 1)
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    stim_duration = params["stim_duration"] #250
    amp = params["amp"] #
    dt = params["dt"] #0.75
    stoptime = params["stoptime"] #70000
    num_shifts = params["num_shifts"] #100
    settle_time = params["settle_time"] #25000
    signals, t = run_model(dt, t_start=0, t_end=1, amp=0, stoptime=stoptime)
    # signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
    # get rid of transients 20000:
    # warning period is in indices not in ms!
    T, T_std = get_period(signals[0, settle_time:])
    # start from the end of expiration (begin of inspiration)
    PreI = signals[0, settle_time:]
    t_start_insp, t_end_insp = (get_insp_starts_and_ends(PreI))
    t_start_insp = (t_start_insp) * dt + settle_time
    t_end_insp = (t_end_insp) * dt + settle_time
    t1_s = t_start_insp[:5]
    # shifts in ms
    shifts = np.array([T * i / num_shifts for i in range(num_shifts)]) * dt

    for i in tqdm(range(len(shifts))[::-1]):
        for j in range(len(t1_s)):
            shift = shifts[i]
            t1 = int(t1_s[j] + shift)
            # print("Shift: {}, Impulse at time : {}".format(shift, t1))
            t2 = t1 + stim_duration
            # create and run a model
            signals, t = run_model(dt, t1, t2, amp, stoptime)
            data = dict()
            data['signals'] = signals
            data['t'] = t
            data['dt'] = dt
            data['phase'] = np.round((2 * np.pi) * (i / len(shifts)), 2)
            data['shift'] = shift
            data['period'] = T
            data['period_std'] = T_std
            data['start_stim'] = t1
            data['duration'] = stim_duration
            data['amp'] = amp
            pickle.dump(data, open(f"{folder_save_to}/run_{amp}_{stim_duration}_{data['phase']}_{j}.pkl", "wb+"))
    return None

if __name__ == '__main__':
    params = {}
    params["dt"] = 0.75
    params["stoptime"] = 65000
    params["num_shifts"] = 50
    params["settle_time"] = 25000
    amps = [200]
    stim_descriptor = 'short'
    stim_durations = [250]
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"
    root_folder_signals = f"{data_path}/num_exp_runs/{stim_descriptor}_stim"
    for stim_duration in stim_durations:
        for amp in amps:
            params["amp"] = amp
            params["stim_duration"] = stim_duration
            print(amp, stim_duration)
            folder_signals = root_folder_signals + f"/num_exp_{stim_descriptor}_stim_{amp}_{stim_duration}"
            create_dir_if_not_exist(folder_signals)
            run_simulations(params, folder_signals)