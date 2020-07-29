import json
import numpy as np
from matplotlib import  pyplot as plt
from num_experiments.Model import Network, NeuralPopulation
from utils.gen_utils import get_project_root


def run_simulations(W, d, dt, amp, stim_start, durations, stoptime):
    data_folder = str(get_project_root()) + "/data"
    img_folder = f"{get_project_root()}/img"
    default_neural_params = json.load(open(f'{data_folder}/params/default_neural_params.json', 'r+'))
    population_names = ['Sw1', 'Sw2', 'Relay']

    Relay = NeuralPopulation("Relay", default_neural_params)
    Sw1 = NeuralPopulation("Sw1", default_neural_params)
    Sw2 = NeuralPopulation("Sw2", default_neural_params)
    Relay.tau_ad = 15000.0
    Sw1.tau_ad = 1000.0
    Sw2.tau_ad = 1000.0

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)
    N = len(population_names)

    drives = d
    for i in range(len(durations)):
        net = Network(populations, W, drives, dt, history_len=int(stoptime / dt))
        net.v = -100 + 100 * np.random.rand(len(population_names))
        inp = np.zeros(net.N)
        inp[population_names.index("Relay")] = amp

        net.run(int(stim_start/ dt))
        # set an impuls input to "three" neuron

        net.set_input_current(inp)
        net.run(int((durations[i]) / dt))

        # run the network further
        net.set_input_current(np.zeros(net.N))
        net.run(int((stoptime - (stim_start + durations[i])) / dt))

        v = np.array(net.v_history)
        fr = net.firing_rate(v, v_half=-30, slope=4)
        fig, axes = plt.subplots(N, 1, figsize=(20,10), sharex=True)
        colors = ['r', 'g', 'k', 'k']
        for j in range(N):
            axes[j].plot(net.t, fr[:, j], color=colors[j], label=population_names[j], linewidth=3)
            axes[j].grid(True)
            axes[j].legend(fontsize=24)
            axes[j].set_ylim([0, 1])
        plt.subplots_adjust(wspace=None, hspace=None)
        img_path = str(get_project_root()) + "/img"
        folder_save_img_to = img_path + "/" + f"other_plots/trigger"
        # fig.savefig(folder_save_img_to + "/" + f"trigger_{stim_start}_{amp}_{durations[i]}" + ".png")
        plt.show()

    return None


if __name__ == '__main__':

    N = 3
    W = np.zeros((N, N))
    W[0, 1] = -0.55 #Sw1 -> Sw2
    W[1, 0] = -0.39 #Sw2 -> Sw1

    W[2, 0] = 0.69 #Relay -> Sw1
    W[2, 1] = 0.71 #Relay -> Sw2

    d = np.array([[0.30, 0.42, 0]]).reshape(1, -1)
    dt = 0.75
    amp = 200
    stim_start = 12500
    durations = [200, 10000]
    stoptime = 35000

    run_simulations(W, d, dt, amp, stim_start, durations, stoptime)




