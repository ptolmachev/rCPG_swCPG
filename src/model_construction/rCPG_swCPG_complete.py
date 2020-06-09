# Comment from Mathias 05.06.2020:
# I actually do not understand what is going on - The VNA looks very nice now and short stimuli resetting looks reasonable.
# We done!
# But now comes the downfall:
# the tonic component during sequential swallowing is still weak -  this could be somehow acceptable -  however!!
# after KF inhibition you still have only a 20-30%  prolongation inspiration as indicated by the phrenic nerve discharge
# and worse is that you maintain postinspiratory discharge - what is published in about 20 papers is complete flat line
# of postinspiratory discharge and doubling of the duration of phrenic nerve discharge as the absolute bare minimum
# - more prolongation would be great.


import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from collections import deque
from src.utils.gen_utils import get_postfix, create_dir_if_not_exist, get_project_root
from src.num_experiments.Model import Network
from src.num_experiments.Model import NeuralPopulation
from datetime import date

def generate_params(inh_NTS, inh_KF):
    params = dict()
    x = [0.1, 1.0, 10][inh_NTS]  # Disinh-inh of NTS
    y = [0.1, 1.0, 10][inh_KF]  # Disinh-inh of KF

    population_names = ['PreI',   # 0
                        'EarlyI', # 1
                        "PostI",  # 2
                        "AugE",   # 3
                        "KF",     # 4
                        "Sw1",    # 5
                        "Sw2",    # 6
                        "Sw3",    # 7
                        "Relay",  # 8
                        "NTS_inh",# 9
                        "KF_inh",  # 10
                        "RampI", # 11
                        "PN",  # 12
                        "HN",  # 13
                        "VN"   # 14
                        ]
    N = len(population_names)

    W = np.zeros((N,N))

    W[0, 1] = 0.40 # PreI -> EarlyI
    W[0, 2] = 0.00 # PreI -> PostI
    W[0, 3] = 0.00 # PreI -> AugE
    W[0, 11] = 0.40 # PreI -> RampI
    W[0, 12] = 0.2 # PreI -> PN
    W[0, 13] = 0.5 # PreI -> HN

    W[1, 0] = -0.08 # EarlyI -> PreI
    W[1, 2] = -0.25 # EarlyI -> PostI
    W[1, 3] = -0.43 # EarlyI -> AugE
    W[1, 5] = -0.01 # EarlyI -> Sw1
    W[1, 11] = -0.20  # EarlyI -> RampI

    W[2, 0] = -0.35 # PostI -> PreI
    W[2, 1] = -0.28 # PostI ->  EarlyI
    W[2, 3] = -0.37 # PostI -> AugE
    W[2, 11] = -0.6  # PostI -> RampI
    W[2, 14] = 0.75 #  PostI -> VN

    W[3, 0] = -0.35 # AugE -> PreI
    W[3, 1] = -0.40 # AugE -> EarlyI
    W[3, 2] = -0.05 # AugE -> PostI
    W[3, 11] = -0.6  # AugE -> RampI

    W[4, 0] = 0.00 # KF -> PreI
    W[4, 2] = 1.30 # KF -> PostI
    W[4, 3] = 0.00 # KF -> AugE
    W[4, 5] = -0.03 # KF -> Sw1
    W[4, 6] = -0.03 # KF -> Sw2

    W[5, 0] = -0.30   # Sw1 -> PreI
    W[5, 1] = -0.30   # Sw1 -> EarlyI
    W[5, 3] = -0.35   # Sw1 -> AugE
    W[5, 14] = 0.4    # Sw1 -> VN
    W[5, 6] = -0.55 * x  # Sw1 -> Sw2
    W[6, 5] = -0.39 * x  # Sw2 -> Sw1

    W[7, 0] = 0.00 # Sw3 -> PreI
    W[7, 2] = 0.80 # Sw3 -> PostI
    # W[7, 3] = 0.15 # Sw3 -> AugE

    W[8, 0] = -0.30  # Relay -> PreI
    W[8, 1] = -0.30  # Relay -> EarlyI
    W[8, 3] = -0.30  # Relay -> AugE
    W[8, 11] = -0.30  # Relay -> RampI
    W[8, 4] = 0.50  # Relay -> KF
    W[8, 5] = 0.69  # Relay -> Sw1
    W[8, 6] = 0.71  # Relay -> Sw2
    W[8, 7] = 0.15  # Relay -> Sw3

    W[9, 5] = -0.1 * x # NTS_inh -> Sw1
    W[9, 6] = -0.1 * x # NTS_inh -> Sw2
    W[9, 7] = -0.1 * x # NTS_inh -> Sw3
    W[9, 8] = -0.1 * x # NTS_inh -> Relay

    W[10, 4] = -0.12 * y # KF_inh -> KF

    W[11, 12] = 0.80 #  RampI -> PN
    W[11, 13] = 0.80 #  RampI -> HN
    W[11, 14] = 0.80 #  RampI -> VN

    drives = np.zeros((5, N))
    # KF
    drives[0, 4] = 0.62 # -> KF

    # NTS
    drives[1, 5] = 0.32  # -> Sw1
    drives[1, 6] = 0.45  # -> Sw2
    drives[1, 7] = 0.62 # -> Sw3

    # other
    drives[2, 0] = 0.10 # -> PreI
    drives[2, 1] = 0.25 # -> EarlyI
    drives[2, 2] = 0.00 # -> PostI
    drives[2, 3] = 0.15 # -> AugE
    drives[2, 9] = 0.3  # -> NTS_inh
    drives[2, 10] = 0.3 # -> KF_inh

    # BotC
    drives[3, 0] = 0.07 # -> PreI
    drives[3, 1] = 0.30 # -> EarlyI
    drives[3, 2] = 0.0 # -> PostI
    drives[3, 3] = 0.4 # -> AugE
    drives[3, 11] = 0.75 # -> RampI

    #PreBotC
    drives[4, 0] = 0.025  # -> PreI

    params["description"] = f"Interaction of Swallowing CPG and rCPG without motor neurons: {str(date.today())}"
    params["W"] = W.tolist()
    params["drives"] = drives.tolist()
    params["populations"] = population_names
    data_path = str(get_project_root()) + "/data"
    json.dump(params, open(f'{data_path}/params/rCPG_swCPG_full_{str(date.today())}.json', 'w', encoding='utf-8'), separators=(',', ':'),
              sort_keys=True, indent=4)
    return None

def construct_model(dt, default_neural_params, connectivity_params):
    population_names = connectivity_params["populations"]
    W = np.array(connectivity_params["W"])
    drives = np.array(connectivity_params["drives"])
    N = len(population_names)
    # create populations
    PreI = NeuralPopulation("PreI", default_neural_params)
    EarlyI = NeuralPopulation("EarlyI", default_neural_params)
    PostI = NeuralPopulation("PostI", default_neural_params)
    AugE = NeuralPopulation("AugE", default_neural_params)
    KF = NeuralPopulation("KF", default_neural_params)
    Sw1 = NeuralPopulation("Sw1", default_neural_params)
    Sw2 = NeuralPopulation("Sw2", default_neural_params)
    Sw3 = NeuralPopulation("Sw3", default_neural_params)
    Relay = NeuralPopulation("Relay", default_neural_params)
    NTS_inh = NeuralPopulation("NTS_inh", default_neural_params)
    KF_inh = NeuralPopulation("KF_inh", default_neural_params)
    RampI = NeuralPopulation("RampI", default_neural_params)
    PN = NeuralPopulation("PN", default_neural_params)
    HN = NeuralPopulation("HN", default_neural_params)
    VN = NeuralPopulation("VN", default_neural_params)

    # modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad = 0.0
    PreI.slope = 8
    PostI.K_ad = 1.3
    PostI.tau_ad = 5000.0
    Relay.tau_ad = 15000.0
    KF.tau_ad = 10000.0
    Sw1.tau_ad = 1000.0
    Sw2.tau_ad = 1000.0
    PN.g_ad = 0.0
    HN.g_ad = 0.0
    VN.g_ad = 0.0

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)
    generate_params(1, 1)
    Network_Model = Network(populations, W, drives, dt, history_len=int(100000/dt))
    return Network_Model

def run_model(net, start, stop, amplitude, duration):
    dt = net.dt
    net.run(int(start / dt))
    # set input to Relay neurons
    inp = np.zeros(net.N)
    inp[8] = amplitude  # Relay Neurons
    net.set_input_current(inp)
    net.run(int((duration) / dt))
    net.set_input_current(np.zeros(net.N))
    net.run(int((stop - (start + duration)) / dt))
    return None

if __name__ == '__main__':
    data_folder = f"{get_project_root()}/data"
    img_folder = f"{get_project_root()}/img"
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json','r'))
    dt = 0.75
    stoptime = 60000
    amp = 200
    # long stim
    stim_duration = 10000
    start = 25000
    create_dir_if_not_exist(img_folder + "/" + f"other_plots/{str(date.today())}")
    for inh_NTS, inh_KF in [(1, 1), (1, 2), (2, 1)]:
        print(amp, stim_duration, start)
        postfix = get_postfix(inh_NTS, inh_KF)
        generate_params(inh_NTS, inh_KF)
        connectivity_params = json.load(open(f'{data_folder}/params/rCPG_swCPG_full_{str(date.today())}.json', 'r'))
        Network_model = construct_model(dt, default_neural_params, connectivity_params)
        run_model(Network_model, start, stoptime, amp, stim_duration)
        fig, axes = Network_model.plot()
        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_swCPG_full_{amp}_{stim_duration}_{postfix}" + ".png")
        plt.close(fig)

    # Short stim:
    stim_duration = 250
    stim_starts = [22000,23000,24000]
    inh_KF = 1
    inh_NTS = 1
    postfix = get_postfix(inh_NTS, inh_KF)
    generate_params(inh_NTS, inh_KF)
    connectivity_params = json.load(open(f'{data_folder}/params/rCPG_swCPG_full_{str(date.today())}.json', 'r'))
    for start in stim_starts:
        print(amp, stim_duration, start)
        Network_model = construct_model(dt, default_neural_params, connectivity_params)
        run_model(Network_model, start, stoptime, amp, stim_duration)
        fig, axes = Network_model.plot()

        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_swCPG_full_{amp}_{stim_duration}_{start}_{postfix}" + ".png")
        plt.close(fig)



