import pickle
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sg
import json
from Model import *
from params_gen import *

def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    plt.plot(series, 'r-',linewidth = 2, alpha = 0.7)
    plt.show()
    plt.close()

def get_postfix(inh_NTS, inh_KF):
    if inh_NTS == 1 and inh_KF == 1:
        postfix = 'intact'
    elif inh_NTS == 2 and inh_KF == 1:
        postfix = 'inh_NTS'
    elif inh_NTS == 1 and inh_KF == 2:
        postfix = 'inh_KF'
    elif inh_NTS == 2 and inh_KF == 2:
        postfix = 'inh_NTS_inh_KF'
    elif inh_NTS == 0 and inh_KF == 1:
        postfix = 'disinh_NTS'
    elif inh_NTS == 1 and inh_KF == 0:
        postfix = 'disinh_KF'
    elif inh_NTS == 0 and inh_KF == 0:
        postfix = 'disinh_NTS_disinh_KF'
    elif inh_NTS == 0 and inh_KF == 2:
        postfix = 'disinh_NTS_inh_KF'
    elif inh_NTS == 2 and inh_KF == 0:
        postfix = 'inh_NTS_disinh_KF'
    return postfix

def get_insp_starts(signals):
    signal_filtered = sg(signals[0], 121, 1)
    threshold = 0.4
    signal_binary = binarise_signal(signal_filtered, threshold)
    signal_change = change(signal_binary)
    begins_inds = find_relevant_peaks(signal_change, 0.5)
    return begins_inds

def get_period(signal):
    # for i in range(len(signals)):
    signal_filtered = sg(signal, 121, 1)
    threshold = np.quantile(signal_filtered, 0.65)
    signal_binary = binarise_signal(signal_filtered, threshold)
    # for correctness check
    # plt.plot(signal_filtered)
    # plt.plot(threshold * np.ones_like(signal_filtered))
    signal_change = change(signal_binary)
    begins = find_relevant_peaks(signal_change, 0.5)
    ends = find_relevant_peaks(-signal_change, 0.5)
    T = np.median(np.hstack([change(begins), change(ends)]))
    std = np.std(np.hstack([change(begins), change(ends)]))
    #Ti =  ends - begins
    # Te - begins - ends

    return T, std

def binarise_signal(signal, threshold):
    res = np.zeros_like(signal)
    res[np.where(signal > threshold)] = 1.0
    res[np.where(signal <= threshold)] = 0
    return res

def change(signal):
    return np.array(signal[1:]) - np.array(signal[:-1])

def last_lesser_than(alist, element):
    #sorted list
    for i in range(len(alist)):
        if alist[i] >= element:
            if i - 1 < 0:
                raise ValueError("The index can't be negative.")
            else:
                return alist[i - 1], i-1
    return [np.nan]

def first_greater_than(alist, element):
    for i in range(len(alist)):
        if alist[i] <= element:
            pass
        else:
            return alist[i], i
    return [np.nan]

def find_relevant_peaks(signal, threshold):
    peaks = scipy.signal.find_peaks(signal)[0]
    return np.array([peaks[i] for i in range(len(peaks)) if abs(signal[peaks[i]]) > threshold])

def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    plt.plot(series, 'r-',linewidth = 2, alpha = 0.7)
    plt.show()
    plt.close()

def get_number_of_breakthroughs(signal, min_amp):
    signal_filtered = sg(signal, 121, 1)[300:]
    threshold = (min_amp)
    signal_binary = binarise_signal(signal_filtered, threshold)
    # for correctness check
    # plt.plot(signal_filtered)
    # plt.plot(threshold * np.ones_like(signal_filtered))
    # identify gaps:
    signal_change = (change(signal_binary))
    signal_begins = np.maximum(0, signal_change)
    signal_ends = np.minimum(0, signal_change)
    # get the indices of jumps
    signal_begins_inds = np.nonzero(signal_begins)[0]
    signal_ends_inds = np.nonzero(signal_ends)[0]
    num_breaks = (len(signal_begins_inds) + len(signal_ends_inds)) / 2

    return num_breaks

def get_features_long_impulse(signals, dt, t_stim_start, t_stim_finish):
    #first one has to cut the relevant signal:
    labels = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2", "Sw3", "KF_t", "KF_p", "KF_r",
              "Motor_HN", "Motor_PN", "Motor_VN", "KF_inh", "NTS_inh"]

    needed_labels = ["PreI", "AugE", "Sw1"]
    ind_stim_start = int(t_stim_start / dt) + 10 # +10 for transients
    ind_stim_finish = int(t_stim_finish / dt)
    signals_relevant = [signals[i][ind_stim_start:ind_stim_finish] for i in range(len(signals)) if labels[i] in needed_labels]

    Sw1 = signals_relevant[needed_labels.index("Sw1")]
    PreI = signals_relevant[needed_labels.index("PreI")]
    AugE = signals_relevant[needed_labels.index("AugE")]

    period, period_std = get_period(Sw1)

    #identifying the number of breakthroughs
    num_swallows = get_number_of_breakthroughs(Sw1, min_amp=0.2)
    num_breakthroughs_PreI = get_number_of_breakthroughs(PreI, min_amp=0.4)
    num_breakthroughs_AugE = get_number_of_breakthroughs(AugE, min_amp=0.1)

    #Rough period estimation:
    if num_swallows != 0:
        period_rough = (t_stim_finish - t_stim_start) / num_swallows
    else:
        period_rough = np.nan
    return period, period_std, period_rough, num_swallows, num_breakthroughs_PreI, num_breakthroughs_AugE


def get_features_short_impulse(signals, dt, t_stim_start, t_stim_finish):
    #first one has to cut the relevant signal:
    labels = ['PreI', 'EarlyI', "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2",
              "Sw3", "KF_t", "KF_p", "KF_relay", "HN", "PN", "VN", "KF_inh", "NTS_inh"]
    needed_labels = ["PreI"]
    signals_relevant = [signals[i] for i in range(len(signals)) if labels[i] in needed_labels]
    PreI = signals_relevant[needed_labels.index("PreI")]
    PreI_filtered = sg(PreI, 121, 1)
    threshold = 0.4 #np.quantile(PreI_filtered[20000: ], 0.65)
    PreI_binary = binarise_signal(PreI_filtered, threshold)

    #get the stimulation time_id
    # stim_id = [peak_id for peak_id in scipy.signal.find_peaks(PostI)[0] if PostI[peak_id] > 0.5][0]
    stim_id = int(t_stim_start / dt)

    PreI_change = change(PreI_binary)
    PreI_begins = find_relevant_peaks(signal=PreI_change, threshold=0.5)
    PreI_ends = find_relevant_peaks(signal=-1.0*PreI_change, threshold=0.5)

    _, i = last_lesser_than(PreI_begins, stim_id)
    begin_id = i - 1 # cause we need one more breathing cycle at the start
    #some margin
    starttime_id = PreI_begins[begin_id] - 500

    stop_peak_id = i + 3
    stoptime_id = PreI_ends[stop_peak_id] + 500

    #discard unnessessary information
    for i in range(len(signals_relevant)):
        signals_relevant[i] = signals_relevant[i][starttime_id:stoptime_id]
    PreI = signals_relevant[needed_labels.index("PreI")]
    PreI_filtered = sg(PreI, 121, 1)
    threshold = 0.4
    PreI_binary = binarise_signal(PreI_filtered, threshold )
    PreI_change = change(PreI_binary)
    PreI_begins = find_relevant_peaks(signal=PreI_change, threshold=0.5)
    PreI_ends = find_relevant_peaks(signal=-PreI_change, threshold=0.5)
    stim_id = stim_id - starttime_id

    ts2 = last_lesser_than(PreI_begins, stim_id)[0]
    ts3 = first_greater_than(PreI_begins, stim_id)[0]
    ts1 = last_lesser_than(PreI_begins, ts2)[0]
    ts4 = first_greater_than(PreI_begins, ts3)[0]

    te1 = first_greater_than(PreI_ends, ts1)[0]
    te2 = first_greater_than(PreI_ends, ts2)[0]
    te3 = first_greater_than(PreI_ends, ts3)[0]
    te4 = first_greater_than(PreI_ends, ts4)[0]

    # plt.plot(PreI_filtered)
    # plt.axvline(ts1, color='k')
    # plt.axvline(ts2, color='r')
    # plt.axvline(ts3, color='g')
    # plt.axvline(ts4, color='b')
    # plt.axvline(te1, color='k')
    # plt.axvline(te2, color='r')
    # plt.axvline(te3, color='g')
    # plt.axvline(te4, color='b')
    # plt.axvline(stim_id, color='m')
    #identifying Ti_0, T0, T1, Phi, Theta (Phi + Theta + delta = T1), Ti_1, Ti_2:
    Ti_0 = (te1 - ts1)*dt
    T0 = (ts2-ts1)*dt
    Phi = (stim_id - ts2) * dt
    Theta = (ts3-stim_id) * dt
    T1 = (ts3 - ts2) * dt
    Ti_1 = (te3 - ts3) * dt
    Ti_2 = (te4 - ts4) * dt
    return Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2

def run_model(t_start, t_end, amp, stoptime, folder_save_to):
    default_neural_params = {
        'C': 20, 'g_NaP': 0.0, 'g_K': 5.0, 'g_ad': 10.0, 'g_l': 2.8, 'g_synE': 10, 'g_synI': 60, 'E_Na': 50,
        'E_K': -85, 'E_ad': -85, 'E_l': -60, 'E_synE': 0, 'E_synI': -75, 'V_half': -30, 'slope': 4, 'tau_ad': 2000,
        'K_ad': 0.9, 'tau_NaP_max': 6000}

    population_names = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2", "Sw3", "KF_t", "KF_p",
                        "KF_r", "HN", "PN", "VN", "KF_inh", "NTS_inh"]

    # create populations
    # for name in population_names:
    #     exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")
    PreI = NeuralPopulation("PreI", default_neural_params)
    EarlyI = NeuralPopulation("EarlyI", default_neural_params)
    PostI = NeuralPopulation("PostI", default_neural_params)
    AugE = NeuralPopulation("AugE", default_neural_params)
    RampI = NeuralPopulation("RampI", default_neural_params)
    Relay = NeuralPopulation("Relay", default_neural_params)
    Sw1 = NeuralPopulation("Sw1", default_neural_params)
    Sw2 = NeuralPopulation("Sw2", default_neural_params)
    Sw3 = NeuralPopulation("Sw3", default_neural_params)
    KF_t = NeuralPopulation("KF_t", default_neural_params)
    KF_p = NeuralPopulation("KF_p", default_neural_params)
    KF_r= NeuralPopulation("KF_r", default_neural_params)
    HN = NeuralPopulation("HN", default_neural_params)
    PN = NeuralPopulation("PN", default_neural_params)
    VN = NeuralPopulation("VN", default_neural_params)
    KF_inh = NeuralPopulation("KF_inh", default_neural_params)
    NTS_inh = NeuralPopulation("NTS_inh", default_neural_params)

    # modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad = HN.g_ad = PN.g_ad = VN.g_ad = 0.0
    HN.g_NaP = PN.g_NaP = VN.g_NaP = 0.0
    Relay.tau_ad = 8000.0

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    inh_NTS = 1
    inh_KF = 1
    generate_params(inh_NTS, inh_KF)
    file = open("../data/rCPG_swCPG.json", "rb+")
    params = json.load(file)
    W = np.array(params["b"])
    drives = np.array(params["c"])
    dt = 0.75
    net = Network(populations, W, drives, dt, history_len=int(stoptime / dt))
    # if for some reason the running has failed try once again
    net.run(int(t_start / dt))
    # set input to Relay neurons
    inp = np.zeros(net.N)
    inp[5] = amp
    net.set_input_current(inp)
    # run for 10 more seconds
    net.run(int((t_end - t_start) / dt))
    net.set_input_current(np.zeros(net.N))
    # run til 60 seconds
    net.run(int((stoptime - (t_end - t_start) - t_start) / dt))


    net.plot(show=False, save_to=f"../img/{folder_save_to}/{amp}.png")
    V_array = net.v_history
    t = np.array(net.t)
    signals = net.firing_rate(V_array, net.V_half, net.slope).T
    return signals, t

if __name__ == '__main__':
    t_start = 10000
    t_end = 20000
    stoptime = 60000
    amp = 0
    folder_save_to = '10_sec_stim_diff_amp'
    signals, t = run_model(t_start, t_end, amp, stoptime, folder_save_to)
    pickle.dump((signals, t), open('../data/signals_intact_model.pkl', 'wb+'))