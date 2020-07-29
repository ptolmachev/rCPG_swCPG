import json
import numpy as np
from model_construction.rCPG_swCPG import construct_model, run_model, set_weights_and_drives
from src.utils.sp_utils import *
import pickle
from tqdm.auto import tqdm
from copy import deepcopy
from num_experiments.params_gen import generate_params
import os
from utils.gen_utils import create_dir_if_not_exist, get_project_root
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def construct_and_run_model(dt, t_start, duration, amp, stoptime):
    data_folder = str(get_project_root()) + "/data"
    img_folder = f"{get_project_root()}/img"
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json', 'r+'))
    population_names = ['PreI', 'EarlyI', "PostI", "AugE", "KF_t", "KF_p", "KF_relay", "NTS_drive",
                        "Sw1", "Sw2", "Relay", "RampI"]
    W, drives = set_weights_and_drives(1, 1, population_names)
    Network_model = construct_model(population_names, W, drives, dt, default_neural_params)
    run_model(Network_model, t_start, stoptime, amp, duration)
    V_array = Network_model.v_history
    t = np.array(Network_model.t)
    signals = Network_model.firing_rate(V_array, Network_model.V_half, Network_model.slope).T
    return signals, t


def run_the_model_with_Relay_const(net, Relay_activity):

    for i in range(T_steps):
        # set activity of a relay neuron
        net.step()

if __name__ == '__main__':
    # sorting out folders
    data_folder = str(get_project_root()) + "/data"
    img_folder = str(get_project_root()) + "/img"
    save_img_folder = f"{img_folder}/num_experiments/varying_Relay"
    save_data_folder = f"{data_folder}/num_exp_runs/varying_Relay"
    create_dir_if_not_exist(save_img_folder)
    create_dir_if_not_exist(save_data_folder)

    # Specifying parameters of tre simulation
    dt = 0.75
    T_steps = int(30000 / dt)
    T_transient = int(10000 / dt)

    #Constructing a model
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json', 'r+'))
    population_names = ['PreI', 'EarlyI', "PostI", "AugE",
                        "KF_t", "KF_p", "KF_relay", "NTS_drive",
                        "Sw1", "Sw2", "Relay", "RampI"]
    W, drives = set_weights_and_drives(1, 1, population_names)
    Network_model = construct_model(population_names, W, drives, dt, default_neural_params)
    Network_model.history_len = T_steps - T_transient
    Network_model.reset()
    Relay_activities = np.linspace(0.01, 0.3, 59)
    # Relay_activities = [0.66]
    for i, Relay_activity in tqdm(enumerate(Relay_activities)):
        Network_model.reset()

        Network_model.freeze_evolution(["Relay"])
        #set the voltage of the Relay neuron to the value corresponding to the specified Relay activity level
        ind = population_names.index("Relay")
        Network_model.v[ind] = \
            Network_model.firing_rate_inv(Relay_activity, Network_model.V_half[ind], Network_model.slope[ind])
        # run the network
        Network_model.run(T_steps)

        #get the variables' history
        v = np.array(Network_model.v_history)[T_transient:]
        m = np.array(Network_model.m_history)[T_transient:]
        h = np.array(Network_model.h_history)[T_transient:]
        t = np.array(Network_model.t)[T_transient:]

        data = dict()
        data['v'] = v
        data['s'] = Network_model.firing_rate(v, Network_model.V_half, Network_model.slope)
        data['h'] = h
        data['m'] = m
        data['t'] = t
        pickle.dump(data, open(f"{save_data_folder}/signals_variable_Relay_lvl_{str.zfill(str(i), 3)}.pkl", "wb+"))

        fig = Network_model.plot()
        plt.suptitle(f"Relay firing rate = {np.round(Relay_activity,4)}", fontsize=28)
        plt.savefig(f"{save_img_folder}/num_experiments/Relay_lvl_{str.zfill(str(i), 3)}.png")
        # plt.show(block = True)
        plt.close()









