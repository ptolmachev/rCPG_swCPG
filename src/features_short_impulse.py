import numpy as np
# from plot_signals import plot_signals
from Model import *
from utils import *
import json
from scipy import signal
from scipy.integrate import odeint
import scipy
from matplotlib import pyplot as plt
import pickle



if __name__ == '__main__':
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])

    t1 = 0
    t2 = 100
    stoptime = 20000
    amp = 0
    signals, t = run_model(t1, t2, amp, stoptime, '100_ms_stim_diff_phase')
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    T, T_std = get_period(signals)

    amp = 450
    t1_s = [(11500+T)+ i*T*(t[-1]/len(t)) for i in range(9)]
    shifts = np.array([T*i/100 for i in range(100)])*(t[-1])/len(t)
    Ti_0s = np.empty((len(shifts), len(t1_s)), dtype = float)
    T0s = np.empty((len(shifts), len(t1_s)), dtype = float)
    T1s = np.empty((len(shifts), len(t1_s)), dtype = float)
    Phis = np.empty((len(shifts), len(t1_s)), dtype = float)
    Thetas = np.empty((len(shifts), len(t1_s)), dtype = float)
    Ti_1s = np.empty((len(shifts), len(t1_s)), dtype = float)
    Ti_2s = np.empty((len(shifts), len(t1_s)), dtype = float)

    for i in range(len(shifts)):
        for j in range(len(t1_s)):
            shift = shifts[i]
            t1 = t1_s[j]+shift
            print("Shift: {}, Impulse at time : {}".format(shift, t1))
            t2 = t1 + 100
            stoptime = 60000
            #create and run a model
            signals, t = run_model(t1, t2, amp, stoptime, '100_ms_stim_diff_phase')
            Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2 = get_features_short_impulse(signals, t)
            Ti_0s[i,j] = Ti_0
            T0s[i,j] = T0
            T1s[i,j] = T1
            Phis[i,j] = Phi
            Thetas[i,j] = Theta
            Ti_1s[i,j] = Ti_1
            Ti_2s[i,j] = Ti_2

    info = dict()
    info['shift'] = shifts
    info['Ti_0s'] = Ti_0s
    info['T0s'] = T0s
    info['T1s'] = T1s
    info['Phis'] = Phis
    info['Thetas'] = Thetas
    info['Ti_1s'] = Ti_1s
    info['Ti_2s'] = Ti_2s
    pickle.dump(info, open('features_var_phase.pkl', 'wb+'))


