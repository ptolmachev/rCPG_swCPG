import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sg
import json

from rCPG_swCPG.src.Model import Network, NeuralPopulation
from rCPG_swCPG.src.params_gen import generate_params


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


def run_model(t_start, t_end, amp, stoptime, folder_save_to):
    default_neural_params = {
        'C': 20,
        'g_NaP': 0.0,
        'g_K': 5.0,
        'g_ad': 10.0,
        'g_l': 2.8,
        'g_synE': 10,
        'g_synI': 60,
        'E_Na': 50,
        'E_K': -85,
        'E_ad': -85,
        'E_l': -60,
        'E_synE': 0,
        'E_synI': -75,
        'V_half': -30,
        'slope': 4,
        'tau_ad': 2000,
        'K_ad': 0.9,
        'tau_NaP_max': 6000}

    population_names = ['PreI',  # 0
                        'EarlyI',  # 1
                        "PostI",  # 2
                        "AugE",  # 3
                        "RampI",  # 4
                        "Relay",  # 5
                        "Sw1",  # 6
                        "Sw2",  # 7
                        "Sw3",  # 8
                        "KF_t",  # 9
                        "KF_p",  # 10
                        "KF_relay",  # 11
                        "HN",  # 12
                        "PN",  # 13
                        "VN",  # 14
                        "KF_inh",  # 15
                        "NTS_inh"]  # 16

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
    KF_relay = NeuralPopulation("KF_relay", default_neural_params)
    HN = NeuralPopulation("HN", default_neural_params)
    PN = NeuralPopulation("PN", default_neural_params)
    VN = NeuralPopulation("VN", default_neural_params)
    KF_inh = NeuralPopulation("KF_inh", default_neural_params)
    NTS_inh = NeuralPopulation("NTS_inh", default_neural_params)

    # modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad = 0.0
    HN.g_NaP = 0.0
    HN.g_ad = 0.0
    PN.g_NaP = 0.0
    PN.g_ad = 0.0
    VN.g_NaP = 0.0
    VN.g_ad = 0.0

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
    dt = 1.0
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
    signals = net.firing_rate(V_array, net.V_half, net.slope)
    return signals, t

if __name__ == '__main__':
    t_start = 10000
    t_end = 20000
    stoptime = 60000
    amp = 0
    folder_save_to = '10_sec_stim_diff_amp'
    run_model(t_start, t_end, amp, stoptime, folder_save_to)