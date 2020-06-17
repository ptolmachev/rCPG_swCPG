import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from collections import deque
from num_experiments.params_gen import generate_params
from src.utils.gen_utils import get_postfix, create_dir_if_not_exist, get_project_root
from src.num_experiments.Model import Network
from src.num_experiments.Model import NeuralPopulation
from tqdm.auto import  tqdm

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

    population_names = ['PreI']
    N = len(population_names)
    #create populations
    for name in population_names:
        exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")

    #modifications:
    PreI.g_NaP = 5
    PreI.g_ad =  0.0
    PreI.slope = 8

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    W = np.zeros((N,N))

    drives = np.zeros((1, N))

    # # Pons
    d = np.linspace(0,0.032,65)
    for i, drive in tqdm(enumerate(d)):
        drives[0, 0] = drive # -> PreI

        dt = 1.0
        net = Network(populations, W, drives, dt, history_len=int(40000/dt))

        net.v = -50 * np.ones(N)# * np.random.rand(N)
        net.h_NaP = 0.4* np.ones(N)# + 0.1 * np.random.rand(N)
        net.m_ad = 0.4* np.ones(N)# + 0.1 * np.random.rand(N)
        # get rid of all transients
        net.run(int(30000/dt)) # runs for 30 seconds
        fig, axes = net.plot()
        fig.suptitle(f"Activity of PreI neurons. Drive = {drive}", fontsize = 24)
        img_path = str(get_project_root()) + "/img"
        create_dir_if_not_exist(f"{img_path}/other_plots/PreI/")
        plt.savefig(f"{img_path}/other_plots/PreI/{i}_drive={np.round(drive,5)}.png")
        # plt.show(block = True)





