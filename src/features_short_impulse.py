import numpy as np
# from plot_signals import plot_signals
from utils import *
import json
from scipy import signal
from scipy.integrate import odeint
import scipy
from matplotlib import pyplot as plt
import pickle
from tqdm.auto import tqdm


if __name__ == '__main__':
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    t1 = 0
    t2 = 100
    dt = 0.75
    stoptime = 75000
    amp = 0
    signals, t = run_model(t1, t2, amp, stoptime, '100ms_stim_diff_phase')
    # signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
    # get rid of transients 20000:
    # warning period is in indices not in ms!
    T, T_std = get_period(signals[0, 25000: ])

    amp = 370
    # start from the end of expiration (begin of inspiration)
    t_start_insp = (get_insp_starts(signals[:, 25000:]) + 25000) * dt
    t1_s = t_start_insp[:9]
    #shifts in ms
    shifts = np.array([T * i / 100 for i in range(100)]) * dt
    stim_duration = 100
    # Ti_0s = np.empty((len(shifts), len(t1_s)), dtype = float)
    # T0s = np.empty((len(shifts), len(t1_s)), dtype = float)
    # T1s = np.empty((len(shifts), len(t1_s)), dtype = float)
    # Phis = np.empty((len(shifts), len(t1_s)), dtype = float)
    # Thetas = np.empty((len(shifts), len(t1_s)), dtype = float)
    # Ti_1s = np.empty((len(shifts), len(t1_s)), dtype = float)
    # Ti_2s = np.empty((len(shifts), len(t1_s)), dtype = float)

    for i in tqdm(range(len(shifts))):
        for j in range(len(t1_s)):
            shift = shifts[i]
            phase = int(np.round(2 * np.pi * (i), 0))
            t1 = int(t1_s[j] + shift)
            # print("Shift: {}, Impulse at time : {}".format(shift, t1))
            t2 = t1 + stim_duration
            stoptime = 70000
            #create and run a model
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
            pickle.dump(data, open(f"../data/short_stim/run_{phase}_{t1}.pkl", "wb+"))

    #         Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2 = get_features_short_impulse(signals, t, t1, t2 )
    #         Ti_0s[i,j] = Ti_0
    #         T0s[i,j] = T0
    #         T1s[i,j] = T1
    #         Phis[i,j] = Phi
    #         Thetas[i,j] = Theta
    #         Ti_1s[i,j] = Ti_1
    #         Ti_2s[i,j] = Ti_2
    #     if i % 1 == 0:
    #         info = dict()
    #         info['shift'] = shifts
    #         info['Ti_0s'] = Ti_0s
    #         info['T0s'] = T0s
    #         info['T1s'] = T1s
    #         info['Phis'] = Phis
    #         info['Thetas'] = Thetas
    #         info['Ti_1s'] = Ti_1s
    #         info['Ti_2s'] = Ti_2s
    #         pickle.dump(info, open('features_var_phase_22_02_2020.pkl', 'wb+'))
    #
    # info = dict()
    # info['shift'] = shifts
    # info['Ti_0s'] = Ti_0s
    # info['T0s'] = T0s
    # info['T1s'] = T1s
    # info['Phis'] = Phis
    # info['Thetas'] = Thetas
    # info['Ti_1s'] = Ti_1s
    # info['Ti_2s'] = Ti_2s
    # pickle.dump(info, open('features_var_phase_22_02_2020.pkl', 'wb+'))


