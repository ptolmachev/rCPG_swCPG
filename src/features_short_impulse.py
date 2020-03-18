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

def run_simulations():
    generate_params(1, 1)
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    t1 = 0
    t2 = 250
    dt = 0.75
    stoptime = 75000
    amp = 0
    signals, t = run_model(t1, t2, amp, stoptime, '100ms_stim_diff_phase')
    # signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
    # get rid of transients 20000:
    # warning period is in indices not in ms!
    T, T_std = get_period(signals[0, 25000:])
    amp = 500
    # start from the end of expiration (begin of inspiration)
    t_start_insp = (get_insp_starts(signals[:, 25000:]) + 25000) * dt
    t1_s = t_start_insp[:9]
    # shifts in ms
    shifts = np.array([T * i / 100 for i in range(100)]) * dt
    stim_duration = 250

    for i in tqdm(range(len(shifts))):
        for j in range(len(t1_s)):
            shift = shifts[i]
            phase = int(np.round(2 * np.pi * (i), 0))
            t1 = int(t1_s[j] + shift)
            # print("Shift: {}, Impulse at time : {}".format(shift, t1))
            t2 = t1 + stim_duration
            stoptime = 70000
            # create and run a model
            signals, t = run_model(t1, t2, amp, stoptime, '100ms_stim_diff_phase')
            data = dict()
            data['signals'] = signals
            data['t'] = t
            data['dt'] = dt
            data['phase'] = int(np.round(2 * np.pi * (i), 0))
            data['shift'] = shift
            data['period'] = T
            data['period_std'] = T_std
            data['start_stim'] = t1
            data['duration'] = stim_duration
            pickle.dump(data, open(f"../data/short_stim/run_{phase}_{j}.pkl", "wb+"))
    return None

def extract_data(path):
    file_names = os.listdir(path)
    info = dict()
    for file_name in file_names:
        file = open(path + "/" + file_name, "rb+")
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
    pickle.dump(info, open("../data/info_var_phase.pkl","wb+"))
    return None


if __name__ == '__main__':
    run_simulations()
    extract_data('../data/short_stim')



