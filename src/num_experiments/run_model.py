import json
import pickle
import numpy as np
from num_experiments.Model import Network, NeuralPopulation
from num_experiments.params_gen import generate_params

def run_model(dt, t_start, t_end, amp, stoptime, folder_save_img_to):
    default_neural_params = {
        'C': 20, 'g_NaP': 0.0, 'g_K': 5.0, 'g_ad': 10.0, 'g_l': 2.8, 'g_synE': 10, 'g_synI': 60, 'E_Na': 50,
        'E_K': -85, 'E_ad': -85, 'E_l': -60, 'E_synE': 0, 'E_synI': -75, 'V_half': -30, 'slope': 4, 'tau_ad': 2000,
        'K_ad': 0.9, 'tau_NaP_max': 6000}

    population_names = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2", "Sw3", "KF_t", "KF_p",
                        "KF_r", "HN", "PN", "VN", "KF_inh", "NTS_inh"]
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
    KF_r= NeuralPopulation("KF_r", default_neural_params)
    HN = NeuralPopulation("HN", default_neural_params)
    PN = NeuralPopulation("PN", default_neural_params)
    VN = NeuralPopulation("VN", default_neural_params)
    KF_inh = NeuralPopulation("KF_inh", default_neural_params)
    NTS_inh = NeuralPopulation("NTS_inh", default_neural_params)

    # modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad = HN.g_ad = PN.g_ad = VN.g_ad = 0.0
    HN.g_NaP = PN.g_NaP = VN.g_NaP = 0.0
    Relay.tau_ad = 8000.0

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    inh_NTS = 1
    inh_KF = 1
    generate_params(inh_NTS, inh_KF)
    file = open("../../data/rCPG_swCPG.json", "rb+")
    params = json.load(file)
    W = np.array(params["b"])
    drives = np.array(params["c"])
    net = Network(populations, W, drives, dt, history_len=int(stoptime / dt))
    # if for some reason the running has failed try once again
    net.run(int(t_start / dt))
    # set input to Relay neurons
    inp = np.zeros(net.N)
    inp[5] = amp # Relay neurons
    net.set_input_current(inp)
    # run for 10 more seconds
    net.run(int((t_end - t_start) / dt))
    net.set_input_current(np.zeros(net.N))
    # run til stoptime
    net.run(int((stoptime - (t_end - t_start) - t_start) / dt))

    net.plot(show=False, save_to=f"../img/{folder_save_img_to}/single_trial_{amp}.png")
    V_array = net.v_history
    t = np.array(net.t)
    signals = net.firing_rate(V_array, net.V_half, net.slope).T
    return signals, t

if __name__ == '__main__':
    dt = 0.75
    t_start = 22000
    t_end = 22500
    stoptime = 60000
    amp = 300
    folder_save_to = 'num_experiments/short_stim'
    signals, t = run_model(dt, t_start, t_end, amp, stoptime, folder_save_to)
    pickle.dump((signals, t), open('../../data/signals_intact_model.pkl', 'wb+'))
