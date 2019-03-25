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

def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    plt.plot(series, 'r-',linewidth = 2, alpha = 0.7)
    plt.show()

def change(signal):
    return np.array([(signal[i+1] - signal[i]) for i in range(len(signal)-1)])

def first_lesser_than(alist, element):
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

def get_features_long_impulse(signals,t):

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

    _, i = first_lesser_than(PreI_begins, stim_id)
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
    t_coef = t[-1] / len(t)

    # print((PreI_peaks_pos- starttime_id)*t_coef)
    # print((PreI_peaks_neg - starttime_id)*t_coef)
    plot_signals(t, signals_relevant, needed_labels, 0, t[-1], filename)

    #identifying Ti_0, T0, T1, Phi, Theta (Phi + Theta + delta = T1), Ti_1, Ti_2:
    Ti_0 = (PreI_ends[0] - PreI_begins[0])*t_coef
    T0 = (PreI_begins[1] - PreI_begins[0])*t_coef
    Phi = (stim_id - first_lesser_than(PreI_begins, stim_id)[0])*t_coef
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
    amp = 450


    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    #two for loops here: one over the shifts, another one over the list of points with the same phase
    t1 = 13000
    print("Amp: {}, time : {}".format(amp, t1))
    t2 = t1 + 100
    stoptime = t2+19900
    res = model(b, c, vectorfield, t1, t2, amp, stoptime)
    t = res[0]
    signals = res[1:]
    labels = ["PreI","EarlyI", "PostI", "AugE", "RampI", "Relay", "NTS1", "NTS2", "NTS3", "KF","Motor_HN", "Motor_PN", "Motor_VN","KF_inh", "NTS_inh"]
    filename = "test"
    Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2 = get_features_long_impulse(signals, t)
    print(Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2)



    # nice_plot(periods_avg)
    # nice_plot(period_std_avg)
    # nice_plot(rough_periods_avg)
    # nice_plot(num_swallows_s_avg)
    # nice_plot(num_breakthroughs_AugE_s_avg)
    # nice_plot(num_breakthroughs_PreI_s_avg)

    # pickle.dump(info, open('features_var_phase.pkl', 'wb+'))


