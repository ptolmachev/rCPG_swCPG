import numpy as np
from plot_signals import plot_signals
from model import *
import json
from scipy import signal
from scipy.integrate import odeint
import scipy
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sg
import pickle

def get_period(signals):
    # for i in range(len(signals)):
    PreI_change = change(signals[0])

    begins = find_relevant_peaks(PreI_change, 0.1)
    T = (np.median([begins[i+1] - begins[i] for i in range(len(begins)-1)] ) )
    std  = (np.std([begins[i+1] - begins[i] for i in range(len(begins)-1)] ) )
    return T, std


def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    plt.plot(series, 'r-',linewidth = 2, alpha = 0.7)
    plt.show()

def change(signal):
    return np.array([(signal[i+1] - signal[i]) for i in range(len(signal)-1)])

def last_lesser_than(alist, element):
    #sorted list
    for i in range(len(alist)):
        if alist[i] >= element:
            if i - 1 < 0:
                raise ValueError("The index can't be negative.")
            else:
                return alist[i - 1], i-1

def first_greater_than(alist, element):
    for i in range(len(alist)):
        if alist[i] <= element:
            pass
        else:
            return alist[i], i
    return None

def find_relevant_peaks(signal, threshold):
    peaks = scipy.signal.find_peaks(signal)[0]
    return np.array([peaks[i] for i in range(len(peaks)) if abs(signal[peaks[i]]) > threshold])

def get_features_short_impulse(signals,t):

    #first one has to cut the relevant signal:
    labels = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "NTS3", "KF", "Motor_HN", "Motor_PN",
              "Motor_VN", "KF_inh", "NTS_inh"]
    needed_labels = ["PreI", "PostI", "AugE", "NTS1"]
    signals_relevant = [signals[i] for i in range(len(signals)) if labels[i] in needed_labels]
    filename = "test"

    PreI = signals_relevant[needed_labels.index("PreI")]
    PostI = signals_relevant[needed_labels.index("PostI")]

    #get the stimulation time
    stim_id = [peak_id for peak_id in scipy.signal.find_peaks(PostI)[0] if PostI[peak_id] > 0.5][0]

    PreI_change = change(PreI)
    PreI_begins = find_relevant_peaks(PreI_change, 0.1)
    PreI_ends = find_relevant_peaks(-1.0*PreI_change, 0.025)

    # # get the boolean variable if the stimulation has occured during the PreI activity
    # Inspiration = False
    # if np.any(np.abs(PreI_ends - stim_id) < 20):
    #     Inspiration = True

    _, i = last_lesser_than(PreI_begins, stim_id)
    begin_id = i - 1 # cause we need one more breathing cycle at the start
    starttime_id = PreI_begins[begin_id]

    stop_peak_id = i + 3
    stoptime_id = PreI_ends[stop_peak_id]

    #discard unnessessary information
    for i in range(len(signals_relevant)):
        signals_relevant[i] = signals_relevant[i][starttime_id:stoptime_id]

    PreI_begins = PreI_begins[begin_id:stop_peak_id]
    PreI_ends = PreI_ends[begin_id:stop_peak_id]
    t = np.array(t[starttime_id:stoptime_id]) - t[starttime_id]
    t_coef = t[1]

    #identifying Ti_0, T0, T1, Phi, Theta (Phi + Theta + delta = T1), Ti_1, Ti_2:
    Ti_0 = (PreI_ends[0] - PreI_begins[0])*t_coef
    T0 = (PreI_begins[1] - PreI_begins[0])*t_coef
    print("T0:", T0)
    Phi = (stim_id - last_lesser_than(PreI_begins, stim_id)[0])*t_coef
    Theta = (first_greater_than(PreI_begins, stim_id)[0] - stim_id)*t_coef
    T1 = Phi + Theta
    Ti_1 = (PreI_ends[-2] - PreI_begins[-2])*t_coef
    Ti_2 = (PreI_ends[-1] - PreI_begins[-1])*t_coef

    return Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2


if __name__ == '__main__':
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])


    t1 = 0
    t2 = 0
    stoptime = 20000
    amp = 0
    signals, t = model(b, c, vectorfield, t1, t2, amp, stoptime)
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    T, T_std = get_period(signals)


    amp = 450
    t1_s = [(11500+T)+ i*T*(t[-1]/len(t)) for i in range(9)]
    shifts = np.array([T*i/100 for i in range(100)])*(t[-1])/len(t)
    # shifts = np.array([2500, 3000])
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
            signals, t = model(b, c, vectorfield, t1, t2, amp, stoptime)
            labels = ["PreI","EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "NTS3", "KF","Motor_HN", "Motor_PN", "Motor_VN","KF_inh", "NTS_inh"]
            # if (i !=10):
            #     plot_signals(t, signals, labels, 0, t[-1], 'test_'+str(i))
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


        # nice_plot(periods_avg)
        # nice_plot(period_std_avg)
        # nice_plot(rough_periods_avg)
        # nice_plot(num_swallows_s_avg)
        # nice_plot(num_breakthroughs_AugE_s_avg)
        # nice_plot(num_breakthroughs_PreI_s_avg)

        # pickle.dump(info, open('features_var_phase.pkl', 'wb+'))


