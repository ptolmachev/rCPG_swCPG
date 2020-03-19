import numpy as np
from utils import *
import json
from scipy import signal
from scipy.integrate import odeint
import scipy
from matplotlib import pyplot as plt
import pickle
from tqdm.auto import tqdm
from params_gen import generate_params
import os

def run_simulations(params, folder_save_to, folder_save_img_to):
    generate_params(1, 1)
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    stim_duration = params["stim_duration"] #250
    amp = params["amp"] #
    dt = params["dt"] #0.75
    stoptime = params["stoptime"] #70000
    num_shifts = params["num_shifts"] #100
    settle_time = params["settle_time"] #25000
    signals, t = run_model(t_start=0, t_end=1, amp=0, stoptime=stoptime, folder_save_img_to=folder_save_img_to)
    # signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
    # get rid of transients 20000:
    # warning period is in indices not in ms!
    T, T_std = get_period(signals[0, settle_time:])
    # start from the end of expiration (begin of inspiration)
    t_start_insp = (get_insp_starts(signals[:, settle_time:]) + settle_time) * dt
    t1_s = t_start_insp[:9]
    # shifts in ms
    shifts = np.array([T * i / num_shifts for i in range(num_shifts)]) * dt

    for i in tqdm(range(len(shifts))):
        for j in range(len(t1_s)):
            shift = shifts[i]
            phase = int(np.round(2 * np.pi * (i), 0))
            t1 = int(t1_s[j] + shift)
            # print("Shift: {}, Impulse at time : {}".format(shift, t1))
            t2 = t1 + stim_duration
            # create and run a model
            signals, t = run_model(t1, t2, amp, stoptime, folder_save_img_to)
            data = dict()
            data['signals'] = signals
            data['t'] = t
            data['dt'] = dt
            data['phase'] = int(np.round(i * 100 /(2 * np.pi), 0))
            data['shift'] = shift
            data['period'] = T
            data['period_std'] = T_std
            data['start_stim'] = t1
            data['duration'] = stim_duration
            data['amp'] = amp
            pickle.dump(data, open(f"{folder_save_to}/run_{amp}_{stim_duration}_{phase}_{j}.pkl", "wb+"))
    return None

def extract_data(signals_path, save_to):
    file_names = os.listdir(signals_path)
    info = dict()
    for file_name in file_names:
        file = open(signals_path + "/" + file_name, "rb+")
        data = pickle.load(file)
        signals = data['signals']
        t = data['t']
        dt = data['dt']
        phase = data['phase']/100
        shift = data['shift']
        T = data['period']
        T_std = data['period_std']
        t1 = data['start_stim']
        stim_duration = data['duration']
        amp = data['amp']
        # if info key already exists, then pass
        if not (phase in list(info.keys())):
            info[phase] = dict()
            info[phase]["Ti0"] = []
            info[phase]["T0"] = []
            info[phase]["T1"] = []
            info[phase]["Phi"] = []
            info[phase]["Theta"] = []
            info[phase]["Ti1"] = []
            info[phase]["Ti2"] = []

        Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2 = \
            get_features_short_impulse(signals, dt, t1, t1 + stim_duration)

        info[phase]["Ti0"].append(Ti_0)
        info[phase]["T0"].append(T0)
        info[phase]["T1"].append(T1)
        info[phase]["Phi"].append(Phi)
        info[phase]["Theta"].append(Theta)
        info[phase]["Ti1"].append(Ti_1)
        info[phase]["Ti2"].append(Ti_2)
        file.close()
    pickle.dump(info, open(f"{save_to}/info_var_phase_{amp}_{stim_duration}.pkl","wb+"))
    return None


if __name__ == '__main__':
    params = {}
    params["stim_duration"] = 250
    stim_duration = params["stim_duration"]
    params["dt"]  = 0.75
    params["stoptime"] = 70000
    params["num_shifts"]  = 3
    params["settle_time"] = 25000
    amps = [100, 200, 300, 400, 500][::-1]
    save_extracted_data_to = "num_exp_results"
    data_path = "../data"
    img_path = "../img"
    for amp in amps:
        params["amp"] = amp
        folder_save_to = f"short_stim_{amp}_{stim_duration}"
        folder_save_img_to = f"100ms_stim_diff_phase_{amp}_{stim_duration}"
        create_dir_if_not_exist(data_path, folder_save_to)
        create_dir_if_not_exist(img_path, folder_save_img_to)
        run_simulations(params, f"{data_path}/{folder_save_to}", f"{img_path}/{folder_save_img_to}")
        extract_data(folder_save_to, f"{data_path}/{save_extracted_data_to}")



