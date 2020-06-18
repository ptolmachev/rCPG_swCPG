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
    if x < 1 :
        postfix += "KF_disinh"
    elif x > 1 :
        postfix += "KF_inh"
    elif x == 1:
        pass
    if y < 1:
        postfix += "_NTS_disinh"
    elif y > 1:
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
    Max_inhibition = 2

    W[p.index("PreI"), p.index("EarlyI")] = 0.40 # PreI -> EarlyI
    W[p.index("PreI"), p.index("PostI")] = 0.00 # PreI -> PostI
    W[p.index("PreI"), p.index("AugE")] = 0.00 # PreI -> AugE

    W[p.index("EarlyI"), p.index("PreI")] = -0.08 # EarlyI -> PreI
    W[p.index("EarlyI"), p.index("PostI")] = -0.25 # EarlyI -> PostI
    W[p.index("EarlyI"), p.index("AugE")] = -0.63 # EarlyI -> AugE
    W[p.index("EarlyI"), p.index("KF_p")] = np.maximum(-Max_inhibition,-0.10 * x) # EarlyI -> KF_p
    W[p.index("EarlyI"), p.index("Sw1")] = np.maximum(-Max_inhibition,-0.001 * y) # EarlyI -> Sw1
    # W[p.index("EarlyI"), p.index("Sw2")] = np.maximum(-Max_inhibition,-0.001 * y) # EarlyI -> Sw2

    W[p.index("PostI"), p.index("PreI")] = -0.32 # PostI -> PreI
    W[p.index("PostI"), p.index("EarlyI")] = -0.20 # PostI -> EarlyI
    W[p.index("PostI"), p.index("AugE")] = -0.36 # PostI -> AugE

    W[p.index("AugE"), p.index("PreI")] = -0.30 # AugE -> PreI
    W[p.index("AugE"), p.index("EarlyI")] = -0.43 # AugE -> EarlyI
    W[p.index("AugE"), p.index("PostI")] = -0.06 # AugE -> PostI

    W[p.index("KF_t"), p.index("PreI")] = +0.16 # KF_t -> PreI
    W[p.index("KF_t"), p.index("EarlyI")] = +0.66 # KF_t -> EarlyI
    W[p.index("KF_t"), p.index("PostI")] = +1.50 # KF_t -> PostI
    W[p.index("KF_t"), p.index("AugE")] = +0.72 # KF_t -> AugE
    W[p.index("KF_t"), p.index("KF_relay")] = +0.7  # KF_t -> KF_relay

    W[p.index("KF_p"), p.index("PreI")] = +0.00 # KF_p -> PreI
    W[p.index("KF_p"), p.index("EarlyI")] = +0.00 # KF_p -> EarlyI
    W[p.index("KF_p"), p.index("PostI")] = +0.00 # KF_p -> PostI
    W[p.index("KF_p"), p.index("AugE")] = +0.00 # KF_p -> AugE

    W[p.index("KF_relay"), p.index("Sw1")] = -0.06  # KF_relay -> Sw1
    W[p.index("KF_relay"), p.index("Sw2")] = -0.05  # KF_relay -> Sw2

    W[p.index("KF_inh"), p.index("KF_t")] = np.maximum(-Max_inhibition,-0.15 * x) # KF_inh -> KF_t
    W[p.index("KF_inh"), p.index("KF_p")] = np.maximum(-Max_inhibition,-0.15 * x) # KF_inh -> KF_p

    W[p.index("NTS_drive"), p.index("PostI")] = 0.42  # NTS_drive -> PostI

    W[p.index("NTS_inh"), p.index("NTS_drive")] = np.maximum(-Max_inhibition,-0.15 * y) # NTS_inh -> NTS_drive
    W[p.index("NTS_inh"), p.index("Relay")] = np.maximum(-Max_inhibition,-0.15 * y)  # NTS_inh -> Relay

    W[p.index("Sw1"), p.index("PreI")] = -0.30   # Sw1 -> PreI
    W[p.index("Sw1"), p.index("EarlyI")] = -0.20   # Sw1 -> EarlyI
    W[p.index("Sw1"), p.index("AugE")] = -0.35   # Sw1 -> AugE
    W[p.index("Sw1"), p.index("Sw2")] = np.maximum(-Max_inhibition,-0.56 * y)  # Sw1 -> Sw2
    W[p.index("Sw2"), p.index("Sw1")] = np.maximum(-Max_inhibition,-0.39 * y)  # Sw2 -> Sw1

    W[p.index("Relay"), p.index("PreI")] = -0.30  # Relay -> PreI
    W[p.index("Relay"), p.index("EarlyI")] = -0.30  # Relay -> EarlyI
    W[p.index("Relay"), p.index("AugE")] = -0.30  # Relay -> AugE
    W[p.index("Relay"), p.index("KF_t")] = 0.50  # Relay -> KF_t
    W[p.index("Relay"), p.index("KF_p")] = 0.50  # Relay -> KF_p
    W[p.index("Relay"), p.index("Sw1")] = 0.69  # Relay -> Sw1
    W[p.index("Relay"), p.index("Sw2")] = 0.71  # Relay -> Sw2
    W[p.index("Relay"), p.index("NTS_drive")] = 0.15  # Relay -> NTS_drive

    drives = np.zeros((3, N))
    # other
    drives[0, p.index("KF_t")] = 1.15  # -> KF_t
    drives[0, p.index("KF_p")] = 0.45  # -> KF_p
    drives[0, p.index("KF_inh")] = 0.60  # -> KF_inh
    drives[0, p.index("NTS_drive")] = 1.00  # -> NTS_drive
    drives[0, p.index("NTS_inh")] = 0.60  # -> NTS_inh
    drives[0, p.index("Sw1")] = 0.33  # -> Sw1
    drives[0, p.index("Sw2")] = 0.45  # -> Sw2

    # BotC
    drives[1, p.index("PreI")] = 0.09 # -> PreI
    drives[1, p.index("EarlyI")] = 0.27 # -> EarlyI
    drives[1, p.index("PostI")] = 0.00 # -> PostI
    drives[1, p.index("AugE")] = 0.42 # -> AugE

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
    NTS_inh = NeuralPopulation("NTS_inh", default_neural_params)
    KF_inh = NeuralPopulation("KF_inh", default_neural_params)

    # modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad = 0.0
    PreI.slope = 8
    PostI.K_ad = 1.3
    PostI.tau_ad = 5000.0
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
    inp[net.population_names.index("Relay")] = amplitude  # Relay Neurons
    net.set_input_current(inp)
    net.run(int((duration) / dt))
    net.set_input_current(np.zeros(net.N))
    net.run(int((stop - (start + duration)) / dt))
    return None


if __name__ == '__main__':
    data_folder = str(get_project_root()) + "/data"
    img_folder = f"{get_project_root()}/img"
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json', 'r+'))

    population_names = ['PreI', 'EarlyI', "PostI", "AugE", "KF_t", "KF_p", "KF_relay", "KF_inh", "NTS_drive", "NTS_inh",
                        "Sw1", "Sw2", "Relay"]
    dt = 0.5
    stoptime = 60000
    amp = 200
    # long stim
    stim_duration = 10000
    start = 25000
    create_dir_if_not_exist(img_folder + "/" + f"other_plots/{str(date.today())}")
    for inh_KF, inh_NTS in [[1, 1], [1, 10], [10, 1]]: #,
        print(amp, stim_duration, start)
        postfix = get_postfix(inh_KF, inh_NTS)
        W, drives = set_weights_and_drives(inh_KF, inh_NTS, population_names)
        Network_model = construct_model(population_names, W, drives, dt, default_neural_params)
        run_model(Network_model, start, stoptime, amp, stim_duration)
        fig, axes = Network_model.plot()
        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_KF_NTS_{amp}_{stim_duration}_{postfix}" + ".png")
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
        fig, axes = Network_model.plot()

        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_KF_NTS_{amp}_{stim_duration}_{start}_{postfix}" + ".png")
        plt.close(fig)



