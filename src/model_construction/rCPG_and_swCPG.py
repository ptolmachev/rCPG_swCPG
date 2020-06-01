import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from collections import deque
from num_experiments.params_gen import generate_params
from src.utils.gen_utils import get_postfix, create_dir_if_not_exist, get_project_root
from src.num_experiments.Model import Network
from src.num_experiments.Model import NeuralPopulation


if __name__ == '__main__':
    default_neural_params = {
    'C' : 20,
    'g_NaP' : 0.0,
    'g_K' : 5.0,
    'g_ad' : 10.0,
    'g_l' : 2.8,
    'g_synE' : 10,
    'g_synI' : 60,
    'E_Na' : 50,
    'E_K' : -85,
    'E_ad' : -85,
    'E_l' : -60,
    'E_synE' : 0,
    'E_synI' : -75,
    'V_half' : -30,
    'slope' : 4,
    'tau_ad' : 2000,
    'K_ad' : 0.9,
    'tau_NaP_max' : 6000}

    population_names = ['PreI',   # 0
                        'EarlyI', # 1
                        "PostI",  # 2
                        "AugE",   # 3
                        "KF",     # 4
                        "Sw1",    # 5
                        "Sw2",    # 6
                        "Sw3",    # 7
                        "Relay",  # 8
                        "NTS_inh" # 9
                        ]
    N = len(population_names)
    #create populations
    for name in population_names:
        exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")

    #modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad =  0.0
    PreI.slope = 8
    PostI.K_ad = 1.3
    Relay.tau_ad = 15000.0
    Sw1.tau_ad = 1000.0
    Sw2.tau_ad = 1000.0
    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    W = np.zeros((N,N))

    W[0, 1] = 0.40 # PreI -> EarlyI
    W[0, 2] = 0.00 # PreI -> PostI
    W[0, 3] = 0.00 # PreI -> AugE

    W[1, 0] = -0.08 # EarlyI -> PreI
    W[1, 2] = -0.25 # EarlyI -> PostI
    W[1, 3] = -0.35 # EarlyI -> AugE

    W[2, 0] = -0.30 # PostI -> PreI
    W[2, 1] = -0.05 # PostI -> EarlyI
    W[2, 3] = -0.24 # PostI -> AugE

    W[3, 0] = -0.20 # AugE -> PreI
    W[3, 1] = -0.35 # AugE -> EarlyI
    W[3, 2] = -0.05 # AugE -> PostI

    W[4, 0] = 0.00 # KF -> PreI
    W[4, 1] = 0.18 # KF -> EarlyI
    W[4, 2] = 1.36 # KF -> PostI
    W[4, 3] = 0.53 # KF -> AugE

    W[4, 0] = 0.00 # KF -> PreI
    W[4, 1] = 0.18 # KF -> EarlyI
    W[4, 2] = 1.36 # KF -> PostI
    W[4, 3] = 0.53 # KF -> AugE


    W[5, 0] = -0.3  # Sw1 -> PreI
    W[5, 1] = -0.3  # Sw1 -> EarlyI
    W[5, 2] = 0.1  # Sw1 -> PostI
    W[5, 6] = -0.55  # Sw1 -> Sw2
    W[6, 5] = -0.39  # Sw2 -> Sw1

    W[7, 0] = 0.00 # Sw3 -> PreI
    W[7, 1] = 0.18 # Sw3 -> EarlyI
    W[7, 2] = 0.88 # Sw3 -> PostI
    W[7, 3] = 0.36 # Sw3 -> AugE

    W[8, 5] = 0.69 # Relay -> Sw1
    W[8, 6] = 0.71 # Relay -> Sw2

    W[9, 5] = -0.1 # NTS_inh -> Sw1
    W[9, 6] = -0.1 # NTS_inh -> Sw2
    W[9, 7] = -0.1 # NTS_inh -> Sw3

    drives = np.zeros((5, N))
    # KF
    drives[0, 4] = 0.6 # -> KF

    # NTS
    drives[1, 5] = 0.33  # -> Sw1
    drives[1, 6] = 0.45  # -> Sw2
    drives[1, 7] = 0.62 # -> Sw3

    # other
    drives[2, 0] = 0.065 # -> PreI
    drives[2, 1] = 0.20 # -> EarlyI
    drives[2, 2] = 0.00 # -> PostI
    drives[2, 3] = 0.08 # -> AugE
    drives[2, 9] = 0.3  # -> NTS_inh

    # BotC
    drives[3, 0] = 0.07 # -> PreI
    drives[3, 1] = 0.30 # -> EarlyI
    drives[3, 2] = 0.0 # -> PostI
    drives[3, 3] = 0.4 # -> AugE

    #PreBotC
    drives[4, 0] = 0.025  # -> PreI

    set_drives = [[1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1]]
    amp =200
    stim_dur = 250
    for i, m in enumerate(set_drives):
        drives_ = (drives.T * np.array(m)).T
        dt = 1.0
        net = Network(populations, W, drives_, dt, history_len=int(40000/dt))

        net.v = -100 * np.random.rand(N)
        net.h_NaP = 0.4 + 0.1 * np.random.rand(N)
        net.m_ad = 0.4 + 0.1 * np.random.rand(N)

        # get rid of all transients
        net.run(int(15000/dt)) # runs for 10 seconds
        inp = np.zeros(net.N)
        inp[8] = amp
        net.set_input_current(inp)
        net.run(int(stim_dur / dt))
        net.set_input_current(np.zeros(net.N))
        net.run(int((30000 - stim_dur - 15000)/dt))

        # run the network further
        net.set_input_current(np.zeros(net.N))
        # net.run(int(3000/dt)) # runs for 30 seconds
        fig, axes = net.plot()
        img_path = str(get_project_root()) + "/img"
        create_dir_if_not_exist(f"{img_path}/other_plots/Rubins_modification/")
        plt.savefig(f"{img_path}/other_plots/Rubins_modification/{i}.png")



