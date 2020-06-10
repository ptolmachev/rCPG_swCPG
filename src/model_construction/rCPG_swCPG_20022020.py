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
    # 0- PreI   # 1 - EarlyI  # 2 - PostI
    # 3 - AugE  # 4 - RampI   # 5 - Relay
    # 6 - Sw 1  # 7 - Sw2     # 8 - Sw3
    # 9 - KF_t   # 10 - KF_p    # 11 - M_HN
    # 12- M_PN  # 13 - M_VN   # 14 - KF_inh
    # 15 - NTS_inh
    population_names = ['PreI',   # 0
                        'EarlyI', # 1
                        "PostI",  # 2
                        "AugE",   # 3
                        "RampI",  # 4
                        "Relay",  # 5
                        "Sw1",    # 6
                        "Sw2",    # 7
                        "Sw3",    # 8
                        "KF_t",   # 9
                        "KF_p",   # 10
                        "KF_relay",#11
                        "HN",     # 12
                        "PN",     # 13
                        "VN",     # 14
                        "KF_inh", # 15
                        "NTS_inh" # 16
                        ]
    N = len(population_names)
    W = np.zeros((N,N))
    W[0, 1] = 0.3  # PreI -> EarlyI
    W[0, 4] = 0.1  # PreI -> RampI
    W[0, 12] = 0.7  # PreI -> HN
    W[0, 13] = 0.2  # PreI -> PN

    W[1, 0] = -0.10  # EarlyI -> PreI
    W[1,2] = -0.3   #EarlyI -> PostI
    W[1,3] = -0.4  #EarlyI -> AugE
    W[1,4] = -0.20  #EarlyI -> RampI
    W[1,10] = -0.3  #EarlyI -> KF_p
    W[1, 6] = -0.02  # EarlyI -> Sw1

    W[2,0] = -0.33    #PostI -> PreI
    W[2,1] = -0.35  #PostI -> EarlyI
    W[2,3] = -0.35   #PostI -> AugE
    W[2,4] = -0.85   #PostI -> RampI
    # W[2,6] = -0.06  #PostI -> Sw1
    # W[2,7] = -0.07  #PostI -> Sw2
    W[2,14] = 0.37  #PostI -> VN

    W[3,0] = -0.50   #AugE -> PreI
    W[3,1] = -0.40  #AugE -> EarlyI
    W[3,2] = -0.02  #AugE -> PostI
    W[3,4] = -0.80  #AugE -> RampI
    # W[3,6] = -0.01 #AugE -> Sw1
    # W[3,7] = -0.02 #AugE -> Sw2
    W[3,9] = -0.01  #AugE -> KF_t
    W[3,10] = -0.01 #AugE -> KF_p

    W[4,12] = 0.2 # RampI -> M_HN
    W[4,13] = 0.7 # RampI -> M_PN
    W[4,14] = 0.60 # RampI -> M_VN

    W[5,0] = -0.40 # Relay -> PreI
    W[5,1] = -0.40 # Relay -> EarlyI
    W[5,2] = 0.50 # Relay -> PostI
    W[5,3] = -0.25 # Relay -> AugE
    W[5,4] = -0.2 # Relay -> RampI
    W[5,6] = 0.69 # Relay -> Sw1
    W[5,7] = 0.71 # Relay -> Sw2
    W[5,8] = 0.65 # Relay -> Sw3
    W[5,9] = 0.5 # Relay -> KF_t
    W[5,10] = 0.5 # Relay -> KF_p

    W[6, 0] = -0.30  # Sw1 -> PreI
    W[6, 1] = -0.30  # Sw1 -> EarlyI
    W[6, 3] = -0.15  # Sw1 -> AugE
    W[6, 4] = -0.35  # Sw1 -> RampI
    W[6, 7] = -0.55 * x #Sw1 -> Sw2
    W[6, 12] = 0.5  # Sw1 -> HN
    W[6, 14] = 0.6  # Sw1 -> VN

    W[7,6] = -0.39 * x #Sw2 -> Sw1

    W[8,1] = 0.2 # Sw3 -> EarlyI
    W[8,2] = 0.55 # Sw3 -> PostI

    W[9, 11] = 1.4  # KF_t -> KF_relay

    W[10,2] = 0.86 # KF_p -> PostI
    W[10,8] = 0.5 # KF_p -> Sw3
    W[10, 14] = 0.38  # KF_p -> M_VN

    W[11,0] = -0.07 #KF_relay -> PreI
    W[11,1] = -0.06 #KF_relay -> EarlyI
    W[11,6] = -0.04 #KF_relay -> Sw1
    # W[11,7] = -0.08 #KF_relay -> Sw2

    W[15,9] = -0.3*y #KF_inh -> KF_t
    W[15,10] = -0.3*y #KF_inh -> KF_p
    W[16,5] = -0.3*x #NTS_inh -> Relay
    W[16,6] = -0.1*x #NTS_inh -> Sw1
    W[16,7] = -0.1*x #NTS_inh -> Sw2
    W[16,8] = -0.2*x #NTS_inh -> Sw3

    drives = np.zeros((1, N))
    # other
    drives[0,0] = 0.265 #To PreI
    drives[0,1] = 0.38  #To EarlyI
    drives[0,2] = 0.05  #To PostI
    drives[0,3] = 0.39  #To AugE
    drives[0,4] = 0.70  #To RampI
    drives[0,6] = 0.33 #To Sw1
    drives[0,7] = 0.45  #To Sw2
    drives[0,8] = 0.8  #To Sw3
    drives[0,9] = 0.8  #To KF_t
    drives[0,10] = 0.8  #To KF_p
    drives[0,11] = 0.0  # To KF_relay
    drives[0,15] = 0.3 #To KF_inh
    drives[0,16] = 0.3 #To NTS_inh


    params["description"] = f"Interaction of Swallowing CPG and rCPG. Model 20022020: {str(date.today())}"
    params["W"] = W.tolist()
    params["drives"] = drives.tolist()
    params["populations"] = population_names
    data_path = str(get_project_root()) + "/data"
    json.dump(params, open(f'{data_path}/params/rCPG_swCPG_20022020_{str(date.today())}.json', 'w', encoding='utf-8'), separators=(',', ':'),
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
    KF_p = NeuralPopulation("KF_p", default_neural_params)
    KF_t = NeuralPopulation("KF_t", default_neural_params)
    KF_relay = NeuralPopulation("KF_relay", default_neural_params)
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

    PreI.g_NaP = 5.0
    PreI.g_ad = HN.g_ad = PN.g_ad = VN.g_ad = RampI.g_ad = 0.0
    HN.g_NaP = PN.g_NaP = VN.g_NaP = 0.0
    Relay.tau_ad = 8000.0
    PostI.tau_ad = 7500.0
    Sw1.tau_ad = 1000.0
    Sw2.tau_ad = 1000.0

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
    inp[5] = amplitude  # Relay Neurons
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
        connectivity_params = json.load(open(f'{data_folder}/params/rCPG_swCPG_20022020_{str(date.today())}.json', 'r'))
        Network_model = construct_model(dt, default_neural_params, connectivity_params)
        run_model(Network_model, start, stoptime, amp, stim_duration)
        fig, axes = Network_model.plot()
        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_swCPG_20022020_{amp}_{stim_duration}_{postfix}" + ".png")
        plt.close(fig)

    # Short stim:
    stim_duration = 250
    stim_starts = [22000,23000,24000]
    inh_KF = 1
    inh_NTS = 1
    postfix = get_postfix(inh_NTS, inh_KF)
    generate_params(inh_NTS, inh_KF)
    connectivity_params = json.load(open(f'{data_folder}/params/rCPG_swCPG_20022020_{str(date.today())}.json', 'r'))
    for start in stim_starts:
        print(amp, stim_duration, start)
        Network_model = construct_model(dt, default_neural_params, connectivity_params)
        run_model(Network_model, start, stoptime, amp, stim_duration)
        fig, axes = Network_model.plot()

        folder_save_img_to = img_folder + "/" + f"other_plots/{str(date.today())}"
        fig.savefig(folder_save_img_to + "/" + f"rCPG_swCPG_20022020_{amp}_{stim_duration}_{start}_{postfix}" + ".png")
        plt.close(fig)



