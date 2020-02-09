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

names = ['PreI',  # 0
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
         "NTS_inh"]  # 015

PreI = NeuralPopulation('PreI', default_neural_params)
EarlyI = NeuralPopulation('EarlyI', default_neural_params)
PostI = NeuralPopulation('PostI', default_neural_params)
AugE = NeuralPopulation('AugE', default_neural_params)
RampI = NeuralPopulation('RampI', default_neural_params)
Relay = NeuralPopulation('Relay', default_neural_params)
Sw1 = NeuralPopulation('Sw1', default_neural_params)
Sw2 = NeuralPopulation('Sw2', default_neural_params)
Sw3 = NeuralPopulation('Sw3', default_neural_params)
KFi = NeuralPopulation('KFi', default_neural_params)
KFe = NeuralPopulation('KFe', default_neural_params)
HN = NeuralPopulation('HN', default_neural_params)
PN = NeuralPopulation('PN', default_neural_params)
VN = NeuralPopulation('VN', default_neural_params)
KF_inh = NeuralPopulation('KF_inh', default_neural_params)
NTS_inh = NeuralPopulation('NTS_inh', default_neural_params)

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
for name in names:
    populations[name] = eval(name)


for inh_NTS in [0, 1, 2]:
    for inh_KF in [0, 1, 2]:
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
        net.plot(show=False, save_to=f"../img/Model_09_02_2020/{get_postfix(inh_NTS, inh_KF)}.png")