import json
import numpy as np
# from Model import *
# from utils import *
# from params_gen import *
from rCPG_swCPG.src.Model import NeuralPopulation, Network
from rCPG_swCPG.src.params_gen import generate_params
from rCPG_swCPG.src.utils import get_postfix

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

population_names = ['PreI',    # 0
         'EarlyI',  # 1
         "PostI",  # 2
         "AugE",  # 3
         "RampI",  # 4
         "Relay",  # 5
         "Sw1",  # 6
         "Sw2",  # 7
         "Sw3",  # 8
         "KFi",  # 9
         "KFe",  # 10
         "HN",  # 11
         "PN",  # 12
         "VN",  # 13
         "KF_inh",  # 14
         "NTS_inh"]  # 15

#define populations
for name in population_names:
    exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")

# modifications:
PreI.g_NaP = 5.0
PreI.g_ad = HN.g_ad = PN.g_ad = VN.g_ad = 0.0
HN.g_NaP = PN.g_NaP = VN.g_NaP = 0.0
Relay.tau_ad = 8000.0
# populations dictionary
populations = dict()
for name in population_names:
    populations[name] = eval(name)


for inh_NTS, inh_KF in [(1,1), (1,2), (2,1)]:
    generate_params(inh_NTS, inh_KF)
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    W = np.array(params["b"])
    drives = np.array(params["c"])
    dt = 1.0
    net = Network(populations, W, drives, dt, history_len=int(40000 / dt))
    # get rid of all transients
    net.run(int(15000 / dt))  # runs for 15 seconds
    # run for 15 more seconds
    net.run(int(15000 / dt))
    # set input to Relay neurons
    inp = np.zeros(net.N)
    inp[5] = 370
    net.set_input_current(inp)
    # run for 10 more seconds
    net.run(int(10000 / dt))
    net.set_input_current(np.zeros(net.N))
    # run for 15 more seconds
    net.run(int(15000 / dt))
    net.plot(show=False, save_to=f"../img/Model_10_02_2020/{get_postfix(inh_NTS, inh_KF)}.png")