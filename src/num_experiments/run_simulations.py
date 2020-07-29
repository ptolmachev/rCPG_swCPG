import json

import numpy as np

from model_construction.rCPG_swCPG import construct_model, run_model, set_weights_and_drives
from src.utils.sp_utils import *
import pickle
from tqdm.auto import tqdm
from copy import deepcopy
from num_experiments.params_gen import generate_params
import os
from utils.gen_utils import create_dir_if_not_exist, get_project_root

def construct_and_run_model(dt, t_start, duration, amp, stoptime):
    data_folder = str(get_project_root()) + "/data"
    img_folder = f"{get_project_root()}/img"
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json', 'r+'))
    population_names = ['PreI', 'EarlyI', "PostI", "AugE", "KF_t", "KF_p", "KF_relay", "NTS_drive",
                        "Sw1", "Sw2", "Relay", "RampI"]
    W, drives = set_weights_and_drives(1, 1, population_names)
    Network_model = construct_model(population_names, W, drives, dt, default_neural_params)
    run_model(Network_model, t_start, stoptime, amp, duration)
    V_array = Network_model.v_history
    t = np.array(Network_model.t)
    signals = Network_model.firing_rate(V_array, Network_model.V_half, Network_model.slope).T
    return signals, t


def run_simulations(params, folder_save_to):
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    stim_duration = params["stim_duration"] #250
    amp = params["amp"] #
    dt = params["dt"] #0.75
    stoptime = params["stoptime"] #70000
    num_shifts = params["num_shifts"] #100
    settle_time_inds = int(params["settle_time"] / dt) #20000 / dt

    signals, t = construct_and_run_model(dt, 0, 1, amp, stoptime)
    # signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
    # get rid of transients 20000:
    # warning period is in indices not in ms!
    # T, T_std = get_period(signals[0, settle_time:])

    # start from the end of expiration (begin of inspiration)
    PreI = signals[0, settle_time_inds:]
    T = get_period(t[settle_time_inds:], PreI)
    t_start_insp_inds, t_end_insp_inds = (get_insp_starts_and_ends(PreI))
    t_start_insp = (t_start_insp_inds + settle_time_inds) * dt
    t_end_insp = (t_end_insp_inds + settle_time_inds) * dt
    t1_s = t_start_insp[:5]
    # shifts in ms
    time_shifts = np.array([T * i / num_shifts for i in range(num_shifts)])
    for i in tqdm(range(len(time_shifts))[::-1]):
        for j in range(len(t1_s)):
            time_shift = time_shifts[i]
            t1 = int(t1_s[j] + time_shift)
            # print("Shift: {}, Impulse at time : {}".format(shift, t1))
            # create and run a model
            signals, t = construct_and_run_model(dt, t1, stim_duration, amp, stoptime) ## t1 and t2 should be specified in ms
            data = dict()
            data['signals'] = signals
            data['t'] = t
            data['dt'] = dt
            data['phase'] = np.round((2 * np.pi) * (i / len(time_shifts)), 2)
            data['period'] = T
            data['start_stim'] = t1
            data['duration'] = stim_duration
            data['amp'] = amp
            pickle.dump(data, open(f"{folder_save_to}/run_{amp}_{stim_duration}_{data['phase']}_{j}.pkl", "wb+"))
    return None

if __name__ == '__main__':
    params = {}
    params["dt"] = 0.75
    params["stoptime"] = 65000
    params["num_shifts"] = 100
    params["settle_time"] = 20000 #ms
    amps = [150, 200]
    stim_descriptor = 'short'
    stim_durations = [250, 500]
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


    # # run a single simulation (for debugging)
    # dt = 0.75
    # t1 =0
    # stim_duration = 1
    # amp = 0
    # stoptime = 60000
    # signals, t = construct_and_run_model(dt, t1, stim_duration, amp, stoptime)  ## t1 and t2 should be specified in ms