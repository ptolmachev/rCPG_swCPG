# June 18 2020

import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from collections import deque
from num_experiments.params_gen import generate_params
from src.utils.gen_utils import get_postfix, create_dir_if_not_exist, get_project_root
from src.num_experiments.Model import Network
from src.num_experiments.Model import NeuralPopulation

def get_postfix(x, y):
    postfix = ''
    if x < 1 :
        postfix += "_KF_disinh"
    elif x > 1 :
        postfix += "_KF_inh"
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
    W = np.zeros((N,N))

    W[p.index("PreI"), p.index("EarlyI")] = 0.40 # PreI -> EarlyI
    W[p.index("PreI"), p.index("PostI")] = 0.00 # PreI -> PostI
    W[p.index("PreI"), p.index("AugE")] = 0.00 # PreI -> AugE

    W[p.index("EarlyI"), p.index("PreI")] = -0.08 # EarlyI -> PreI
    W[p.index("EarlyI"), p.index("PostI")] = -0.25 # EarlyI -> PostI
    W[p.index("EarlyI"), p.index("AugE")] = -0.63 # EarlyI -> AugE
    W[p.index("EarlyI"), p.index("KF_p")] = -0.63 # EarlyI -> KF_p

    W[p.index("PostI"), p.index("PreI")] = -0.32 # PostI -> PreI
    W[p.index("PostI"), p.index("EarlyI")] = -0.20 # PostI -> EarlyI
    W[p.index("PostI"), p.index("AugE")] = -0.36 # PostI -> AugE

    W[p.index("AugE"), p.index("PreI")] = -0.30 # AugE -> PreI
    W[p.index("AugE"), p.index("EarlyI")] = -0.43 # AugE -> EarlyI
    W[p.index("AugE"), p.index("PostI")] = -0.06 # AugE -> PostI

    W[p.index("KF_t"), p.index("PreI")] = +0.16 # KF_t -> PreI
    W[p.index("KF_t"), p.index("EarlyI")] = +0.66 # KF_t -> EarlyI
    W[p.index("KF_t"), p.index("PostI")] = +1.55 # KF_t -> PostI
    W[p.index("KF_t"), p.index("AugE")] = +0.72 # KF_t -> AugE

    W[p.index("KF_p"), p.index("PreI")] = +0.00 # KF_p -> PreI
    W[p.index("KF_p"), p.index("EarlyI")] = +0.00 # KF_p -> EarlyI
    W[p.index("KF_p"), p.index("PostI")] = +0.00 # KF_p -> PostI
    W[p.index("KF_p"), p.index("AugE")] = +0.00 # KF_p -> AugE

    W[p.index("KF_inh"), p.index("KF_t")] = -0.15 * x # KF_inh -> KF_t
    W[p.index("KF_inh"), p.index("KF_p")] = -0.15 * x # KF_inh -> KF_p

    W[p.index("NTS_drive"), p.index("PostI")] = 0.40  # NTS_drive -> PostI

    W[p.index("NTS_inh"), p.index("NTS_drive")] = -0.15 * y  # NTS_inh -> NTS_drive

    W[p.index("Sw1"), p.index("PreI")] = -0.30   # Sw1 -> PreI
    W[p.index("Sw1"), p.index("EarlyI")] = -0.30   # Sw1 -> EarlyI
    W[p.index("Sw1"), p.index("AugE")] = -0.35   # Sw1 -> AugE
    W[p.index("Sw1"), p.index("Sw2")] = -0.55 * x  # Sw1 -> Sw2
    W[p.index("Sw2"), p.index("Sw1")] = -0.39 * x  # Sw2 -> Sw1

    drives = np.zeros((3, N))
    # other
    drives[0, p.index("KF_t")] = 1.15  # -> KF_t
    drives[0, p.index("KF_p")] = 1.15  # -> KF_p
    drives[0, p.index("KF_inh")] = 0.60  # -> KF_inh
    drives[0, p.index("NTS_drive")] = 1.00  # -> NTS_drive
    drives[0, p.index("NTS_inh")] = 0.60  # -> NTS_inh
    drives[0, p.index("Sw1")] = 0.32  # -> Sw1
    drives[0, p.index("Sw2")] = 0.45  # -> Sw2


    # BotC
    drives[1, p.index("PreI")] = 0.09 # -> PreI
    drives[1, p.index("EarlyI")] = 0.27 # -> EarlyI
    drives[1, p.index("PostI")] = 0.00 # -> PostI
    drives[1, p.index("AugE")] = 0.42 # -> AugE

    #PreBotC
    drives[2, p.index("PreI")] = 0.025  # -> PreI
    return W, drives

if __name__ == '__main__':
    data_folder = str(get_project_root()) + "/data"
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json', 'r+'))

    population_names = ['PreI', 'EarlyI', "PostI", "AugE", "KF_t", "KF_p", "KF_inh", "NTS_drive", "NTS_inh",
                        "Sw1", "Sw2", "Relay"]
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

    # x = 0.1 - disinhibition
    # x = 10 - inhibition
    xs = [(0.02, 1), (1, 1), (10, 1), (1, 10)]

    for i, tpl in enumerate(xs):
        W, drives = set_weights_and_drives(*tpl, population_names)
        dt = 0.75
        net = Network(populations, W, drives, dt, history_len=int(40000/dt))

        net.v = -100 * np.random.rand(N)
        net.h_NaP = 0.4 + 0.1 * np.random.rand(N)
        net.m_ad = 0.4 + 0.1 * np.random.rand(N)

        # get rid of all transients
        net.run(int(30000/dt)) # runs for 30 seconds
        # print(net.drives[1, :] / net.firing_rate(np.array(net.v_history)[23000, 7], net.V_half, net.slope))

        fig, axes = net.plot()
        img_path = str(get_project_root()) + "/img"
        create_dir_if_not_exist(f"{img_path}/other_plots/rCPG_KF_NTS/")
        plt.savefig(f"{img_path}/other_plots/rCPG_KF_NTS/{get_postfix(*tpl)}.png")
        # plt.show(block = True)



