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

def get_period(signals):
    # for i in range(len(signals)):
    PreI_change = change(signals[0])
    begins = find_relevant_peaks(PreI_change, 0.1)
    T = (np.median([begins[i+1] - begins[i] for i in range(len(begins)-1)] ) )
    std = (np.std([begins[i+1] - begins[i] for i in range(len(begins)-1)] ) )
    return T, std


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

def nice_plot(series):
    fig = plt.figure(figsize = (16,4))
    plt.grid(True)
    plt.plot(series, 'r-',linewidth = 2, alpha = 0.7)
    plt.show()
    plt.close()

def get_number_of_breakthroughs(signal, min_span):
    signal_filtered = sg(signal, 121, 1)
    signal_binary = np.zeros_like(signal_filtered)
    # for correctness check
    # plt.plot(signal_filtered)
    # plt.plot(np.quantile(signal_filtered, 0.7) * np.ones_like(signal_filtered))
    signal_binary[np.where(signal > np.quantile(signal_filtered, 0.9))] = 1.0
    signal_binary[np.where(signal <= np.quantile(signal_filtered, 0.9))] = 0.0
    # identify gaps:
    signal_change = np.abs(signal_binary[1:] - signal_binary[:-1])
    # get the indices of jumps
    signal_change_inds  = np.nonzero(signal_change)[0]
    if len(signal_change_inds) == 0:
        num_breaks = 0
    elif len(signal_change_inds) == 1:
        num_breaks = 0.5
    else:
        # if these jumps are too close - discard
        num_breaks = (np.sum(1.0 * (signal_change_inds[1:] - signal_change_inds[:-1] > min_span)) + 1) / 2
    return num_breaks

def get_features_long_impulse(signals, t, t_stim_start, t_stim_finish):
    #first one has to cut the relevant signal:
    labels = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2", "Sw3", "KF_t", "KF_p", "KF_r",
              "Motor_HN", "Motor_PN", "Motor_VN", "KF_inh", "NTS_inh"]

    needed_labels = ["PreI", "AugE", "Sw1"]
    ind_stim_start = np.where(np.array(t) <= t_stim_start)[0][-1]
    ind_stim_finish = np.where(np.array(t) <= t_stim_finish)[0][-1]
    t = np.array(t[ind_stim_start:ind_stim_finish]) - t[ind_stim_start]
    signals_relevant = [signals[i][ind_stim_start:ind_stim_finish] for i in range(len(signals)) if labels[i] in needed_labels]

    Sw1 = signals_relevant[needed_labels.index("Sw1")]
    PreI = signals_relevant[needed_labels.index("PreI")]
    AugE = signals_relevant[needed_labels.index("AugE")]

    #identifying instantaneous frequency:
    corr = sg(scipy.signal.correlate(Sw1, Sw1,'same'), 121, 1)
    peaks = scipy.signal.find_peaks(corr)[0]

    if len(peaks) >= 3:
        period = np.mean([peaks[i] - peaks[i - 1] for i in range(1, len(peaks))]) * (t_stim_finish - t_stim_start) / len(t)
        period_std = np.std([peaks[i] - peaks[i - 1] for i in range(1, len(peaks))]) * (t_stim_finish - t_stim_start) / len(t)
    else:
        period = np.inf
        period_std = 0

    #identifying the number of breakthroughs
    num_swallows = get_number_of_breakthroughs(Sw1, 50)
    num_breakthroughs_PreI  = get_number_of_breakthroughs(PreI, 50)
    num_breakthroughs_AugE  = get_number_of_breakthroughs(AugE, 50)

    #Rough period estimation:
    if num_swallows != 0:
        period_rough = (t_stim_finish - t_stim_start) / num_swallows
    else:
        period_rough = np.inf
    return period, period_std, period_rough, num_swallows, num_breakthroughs_PreI, num_breakthroughs_AugE


def get_features_short_impulse(signals, t, t_stim_finish, t_stim_start):
    #first one has to cut the relevant signal:
    labels = ['PreI', 'EarlyI', "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2",
              "Sw3", "KF_t", "KF_p", "KF_relay", "HN", "PN", "VN", "KF_inh", "NTS_inh"]
    needed_labels = ["PreI", "PostI"]
    signals_relevant = [signals[i] for i in range(len(signals)) if labels[i] in needed_labels]
    PreI = signals_relevant[needed_labels.index("PreI")]
    PostI = signals_relevant[needed_labels.index("PostI")]

    #get the stimulation time_id
    # stim_id = [peak_id for peak_id in scipy.signal.find_peaks(PostI)[0] if PostI[peak_id] > 0.5][0]
    stim_id = t.tolist().index(t_stim_start)

    PreI_change = change(PreI)
    #TODO check if it really works
    PreI_begins = find_relevant_peaks(signal=PreI_change, threshold=0.1)
    PreI_ends = find_relevant_peaks(signal=-1.0*PreI_change, threshold=0.025)

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
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    W = np.array(params["b"])
    drives = np.array(params["c"])
    dt = 0.75
    net = Network(populations, W, drives, dt, history_len=int(stoptime / dt))
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