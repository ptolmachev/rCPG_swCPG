# June 18 2020
from datetime import date

import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from collections import deque
from src.utils.gen_utils import create_dir_if_not_exist, get_project_root
from src.num_experiments.Model import Network
from src.num_experiments.Model import NeuralPopulation

def get_postfix(x, y):
    postfix = ''
    if x == 2 :
        postfix += "KF_disinh"
    elif x == 0 :
        postfix += "KF_inh"
    elif x == 1:
        pass
    if y == 2:
        postfix += "_NTS_disinh"
    elif y == 0:
        postfix += "_NTS_inh"
    elif y == 1:
        pass

    if x == 1 and y== 1:
        postfix = 'intact'
    return postfix

def set_weights_and_drives(x, y, population_names):
    p = population_names
    N = len(p)
    W = np.zeros((N,N))

    W[p.index("PreI"), p.index("EarlyI")] = 0.40 # PreI -> EarlyI
    W[p.index("PreI"), p.index("PostI")] = 0.00 # PreI -> PostI
    W[p.index("PreI"), p.index("AugE")] = 0.00 # PreI -> AugE
    W[p.index("PreI"), p.index("RampI")] = 0.90 # PreI -> RampI

    W[p.index("EarlyI"), p.index("PreI")] = -0.08 # EarlyI -> PreI
    W[p.index("EarlyI"), p.index("PostI")] = -0.25 # EarlyI -> PostI
    W[p.index("EarlyI"), p.index("AugE")] = -0.63 # EarlyI -> AugE
    W[p.index("EarlyI"), p.index("KF_p")] = -0.10 # EarlyI -> KF_p
    W[p.index("EarlyI"), p.index("Sw1")] = -0.003 # EarlyI -> Sw1
    W[p.index("EarlyI"), p.index("RampI")] = -0.15 # EarlyI -> RampI

    W[p.index("PostI"), p.index("PreI")] = -0.35 # PostI -> PreI
    W[p.index("PostI"), p.index("EarlyI")] = -0.22 # PostI -> EarlyI
    W[p.index("PostI"), p.index("AugE")] = -0.36 # PostI -> AugE
    W[p.index("PostI"), p.index("RampI")] = -0.50 # PostI -> RampI

    W[p.index("AugE"), p.index("PreI")] = -0.30 # AugE -> PreI
    W[p.index("AugE"), p.index("EarlyI")] = -0.43 # AugE -> EarlyI
    W[p.index("AugE"), p.index("PostI")] = -0.06 # AugE -> PostI
    W[p.index("AugE"), p.index("RampI")] = -0.50 # AugE -> RampI

    W[p.index("KF_t"), p.index("PreI")] = +0.16 * x # KF_t -> PreI
    W[p.index("KF_t"), p.index("EarlyI")] = +0.66 * x # KF_t -> EarlyI
    W[p.index("KF_t"), p.index("PostI")] = +1.10 * x # KF_t -> PostI
    W[p.index("KF_t"), p.index("AugE")] = +0.72 * x # KF_t -> AugE
    W[p.index("KF_t"), p.index("KF_relay")] = +0.7 * x  # KF_t -> KF_relay

    W[p.index("KF_p"), p.index("PreI")] = +0.00 * x # KF_p -> PreI
    W[p.index("KF_p"), p.index("EarlyI")] = +0.00 * x# KF_p -> EarlyI
    W[p.index("KF_p"), p.index("PostI")] = +0.60 * x# KF_p -> PostI
    W[p.index("KF_p"), p.index("AugE")] = +0.00 * x# KF_p -> AugE

    W[p.index("KF_relay"), p.index("Sw1")] = -0.09  # KF_relay -> Sw1
    W[p.index("KF_relay"), p.index("Sw2")] = -0.05  # KF_relay -> Sw2

    W[p.index("NTS_drive"), p.index("PostI")] = 0.42  # NTS_drive -> PostI


    W[p.index("Sw1"), p.index("PreI")] = -0.30   # Sw1 -> PreI
    W[p.index("Sw1"), p.index("EarlyI")] = -0.17   # Sw1 -> EarlyI
    W[p.index("Sw1"), p.index("AugE")] = -0.15   # Sw1 -> AugE
    W[p.index("Sw1"), p.index("Sw2")] = -0.56  # Sw1 -> Sw2
    W[p.index("Sw2"), p.index("Sw1")] = -0.39  # Sw2 -> Sw1

    W[p.index("Relay"), p.index("PreI")] = -0.30 * y # Relay -> PreI
    W[p.index("Relay"), p.index("EarlyI")] = -0.30 * y  # Relay -> EarlyI
    W[p.index("Relay"), p.index("AugE")] = -0.30 * y # Relay -> AugE
    W[p.index("Relay"), p.index("RampI")] = -0.30 * y # Relay -> RampI
    W[p.index("Relay"), p.index("KF_t")] = 0.15 * x * y  # Relay -> KF_t
    W[p.index("Relay"), p.index("KF_p")] = 0.15 * x * y # Relay -> KF_p
    W[p.index("Relay"), p.index("Sw1")] = 0.74 * y  # Relay -> Sw1
    W[p.index("Relay"), p.index("Sw2")] = 0.71* y  # Relay -> Sw2
    W[p.index("Relay"), p.index("NTS_drive")] = 0.15 * y  # Relay -> NTS_drive



    drives = np.zeros((3, N))
    # other
    drives[0, p.index("KF_t")] = 0.81 * x  # -> KF_t
    drives[0, p.index("KF_p")] = 0.50 * x  # -> KF_p
    drives[0, p.index("NTS_drive")] = 0.68 * y  # -> NTS_drive
    drives[0, p.index("Sw1")] = 0.33 * y  # -> Sw1
    drives[0, p.index("Sw2")] = 0.45 * y  # -> Sw2

    # BotC
    drives[1, p.index("PreI")] = 0.09 # -> PreI
    drives[1, p.index("EarlyI")] = 0.27 # -> EarlyI
    drives[1, p.index("PostI")] = 0.00 # -> PostI
    drives[1, p.index("AugE")] = 0.42 # -> AugE
    drives[1, p.index("RampI")] = 0.50 # -> RampI

    #PreBotC
    drives[2, p.index("PreI")] = 0.025  # -> PreI
    return W, drives

def construct_model(population_names, W, drives, dt, default_neural_params):
    N = len(population_names)
    # create populations
    PreI = NeuralPopulation("PreI", default_neural_params)
    EarlyI = NeuralPopulation("EarlyI", default_neural_params)
    PostI = NeuralPopulation("PostI", default_neural_params)
    AugE = NeuralPopulation("AugE", default_neural_params)
    KF_t = NeuralPopulation("KF_t", default_neural_params)
    KF_p = NeuralPopulation("KF_p", default_neural_params)
    KF_relay = NeuralPopulation("KF_relay", default_neural_params)
    Sw1 = NeuralPopulation("Sw1", default_neural_params)
    Sw2 = NeuralPopulation("Sw2", default_neural_params)
    NTS_drive = NeuralPopulation("NTS_drive", default_neural_params)
    Relay = NeuralPopulation("Relay", default_neural_params)
    RampI = NeuralPopulation("RampI", default_neural_params)

    # modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad = 0.0
    PreI.slope = 8
    PostI.K_ad = 1.3
    PostI.tau_ad = 5000.0
    KF_p.tau_ad = 6000.0
    Relay.tau_ad = 15000.0
    Sw1.tau_ad = 1000.0
    Sw2.tau_ad = 1000.0

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    Network_Model = Network(populations, W, drives, dt, history_len=int(100000/dt))
    return Network_Model

def run_model(net, start, stop, amplitude, duration):
    dt = net.dt
    net.run(int(start / dt))
    # set input to Relay neurons
    inp = np.zeros(net.N)
    inp[net.pop_names.index("Relay")] = amplitude  # Relay Neurons
    net.set_input_current(inp)
    net.run(int((duration) / dt))
    net.set_input_current(np.zeros(net.N))
    net.run(int((stop - (start + duration)) / dt))
    return None

def plot(net):
    start = 10000
    V_array = np.array(net.v_history).T
    t_array = np.array(net.t)
    N = net.N
    fig, axes = plt.subplots(N+3, 1, figsize=(25, 15))
    if type(axes) != np.ndarray: axes = [axes]
    fr = net.firing_rate(V_array.T, net.V_half, net.slope).T
    for i in range(net.N):
        if i == 0: axes[i].set_title('Firing Rates', fontdict={"size" : 20})
        axes[i].plot(t_array[start:], fr[i, start:], 'k', linewidth=3, label=str(net.names[i]), alpha=0.9)
        axes[i].legend(loc = 1, fontsize=25)
        axes[i].set_ylim([-0.0, 1.0])
        axes[i].set_yticks([])
        axes[i].set_yticklabels([])
        if i != len(axes) - 1:
            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
        axes[i].set_xlabel('t, ms', fontdict={"size" : 20})

    PNA = 0.1 * fr[net.names.index("PreI"), start:] + 0.9 * fr[net.names.index("RampI"), start:]
    HNA = 0.7 * fr[net.names.index("PreI"), start:] + 0.15 * fr[net.names.index("RampI"), start:] \
          + 0.35 * fr[net.names.index("Sw1"),start:]
    VNA = 0.75 * fr[net.names.index("RampI"), start:] + 0.8 * fr[net.names.index("Sw1"), start:] + \
          0.6 * fr[net.names.index("PostI"), start:] + 0.4 * fr[net.names.index("KF_p"), start:]
    Motor_outputs = [PNA, HNA, VNA]
    motor_outputs_names = ["PNA", "HNA", "VNA"]
    for i in range(N, N+3):
        axes[i].plot(t_array[start:], Motor_outputs[i - N], 'k', linewidth=3, label=str(motor_outputs_names[i - N]), alpha=0.9)
        axes[i].legend(loc=1, fontsize=20)
        axes[i].set_ylim([-0.0, 1.0])
        axes[i].set_yticks([])
        axes[i].set_yticklabels([])
        if i != len(axes) - 1:
            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
        axes[i].set_xlabel('t, ms', fontdict={"size": 20})
    plt.subplots_adjust(wspace=0.01, hspace=0)
    return fig, axes

if __name__ == '__main__':
    data_folder = str(get_project_root()) + "/data"
    img_folder = f"{get_project_root()}/img"
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json', 'r+'))

    population_names = ['PreI', 'EarlyI', "PostI", "AugE", "KF_t", "KF_p", "KF_relay", "NTS_drive",
                        "Sw1", "Sw2", "Relay", "RampI"]
    dt = 0.5
    stoptime = 60000
    amp = 150
    # long stim
    stim_duration = 10000
    start = 25000
    create_dir_if_not_exist(img_folder + "/" + f"other_plots/{str(date.today())}")
    for inh_KF, inh_NTS in [[1, 1], [1, 0], [0, 1]]: #,
        print(amp, stim_duration, start)
        postfix = get_postfix(inh_KF, inh_NTS)
        W, drives = set_weights_and_drives(inh_KF, inh_NTS, population_names)
        Network_model = construct_model(population_names, W, drives, dt, default_neural_params)
        run_model(Network_model, start, stoptime, amp, stim_duration)

        fig, axes = plot(Network_model)
        # fig, axes = Network_model.plot()
        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_swCPG_{amp}_{stim_duration}_{postfix}" + ".png")
        plt.close(fig)

    # Short stim:
    stim_duration = 250
    stim_starts = [22000,23000,24000]
    inh_KF = 1
    inh_NTS = 1
    postfix = get_postfix(inh_KF, inh_NTS)
    W, drives = set_weights_and_drives(inh_KF, inh_NTS, population_names)
    for start in stim_starts:
        print(amp, stim_duration, start)
        Network_model = construct_model(population_names, W, drives, dt, default_neural_params)
        run_model(Network_model, start, stoptime, amp, stim_duration)
        # fig, axes = Network_model.plot()
        fig, axes = plot(Network_model)
        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_swCPG_{amp}_{stim_duration}_{start}_{postfix}" + ".png")
        plt.close(fig)



