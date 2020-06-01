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
                        "AugE"]   # 3
    N = len(population_names)
    #create populations
    for name in population_names:
        exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")

    #modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad =  0.0
    PreI.slope = 8

    PostI.K_ad = 1.3
    # PostI.tau_ad = 1000.0


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

    drives = np.zeros((5, N))

    # # Pons
    drives[0, 0] = 0.00 # -> PreI
    drives[0, 1] = 0.05 # -> EarlyI
    drives[0, 2] = 0.38 # -> PostI
    drives[0, 3] = 0.15 # -> AugE

    # # NTS
    drives[1, 0] = 0.00 # -> PreI
    drives[1, 1] = 0.05 # -> EarlyI
    drives[1, 2] = 0.25 # -> PostI
    drives[1, 3] = 0.1 # -> AugE

    # other
    drives[2, 0] = 0.065 # -> PreI
    drives[2, 1] = 0.20 # -> EarlyI
    drives[2, 2] = 0.00 # -> PostI
    drives[2, 3] = 0.08 # -> AugE

    # BotC
    drives[3, 0] = 0.07 # -> PreI
    drives[3, 1] = 0.30 # -> EarlyI
    drives[3, 2] = 0.0 # -> PostI
    drives[3, 3] = 0.4 # -> AugE

    #PreBotC
    drives[4, 0] = 0.025  # -> PreI

    set_drives = [[1,1,1,1,1],[0,1,1,1,1] ,[1,0,1,1,1], [0,0,0,1,1]]
    for i, m in enumerate(set_drives):
        drives_ = (drives.T * np.array(m)).T
        dt = 1.0
        net = Network(populations, W, drives_, dt, history_len=int(40000/dt))

        net.v = -100 * np.random.rand(N)
        net.h_NaP = 0.4 + 0.1 * np.random.rand(N)
        net.m_ad = 0.4 + 0.1 * np.random.rand(N)

        # get rid of all transients
        net.run(int(30000/dt)) # runs for 30 seconds
        fig, axes = net.plot()
        img_path = str(get_project_root()) + "/img"
        create_dir_if_not_exist(f"{img_path}/other_plots/Rubins_modification/")
        plt.savefig(f"{img_path}/other_plots/Rubins_modification/{i}.png")
        # plt.show(block = True)




    # Params as in rubins model
    # # Pons
    # drives[0, 0] = 0.115 # -> PreI
    # drives[0, 1] = 0.30 # -> EarlyI
    # drives[0, 2] = 0.63 # -> PostI
    # drives[0, 3] = 0.33 # -> AugE
    #
    # # BotC
    # drives[2, 0] = 0.07 # -> PreI
    # drives[2, 1] = 0.30 # -> EarlyI
    # drives[2, 2] = 0.0 # -> PostI
    # drives[2, 3] = 0.4 # -> AugE
    #
    # #PreBotC
    # drives[3, 0] = 0.025  # -> PreI

