import numpy as np
from matplotlib import  pyplot as plt
from num_experiments.Model import Network, NeuralPopulation

if __name__ == '__main__':
    default_neural_params = {
    'C' : 20,
    'g_NaP' : 0.0,
    'g_K' : 5.0,
    'g_ad' : 10.0,
    'g_l' : 2.8,
    'g_synE' : 80,
    'g_synI' : 80,
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
    'tau_NaP_max' : 6000,
    'tau_synE' : 10,
    'tau_synI' : 10}

    population_names = ['one']#, 'two']

    #create populations
    for name in population_names:
        exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    N = len(population_names)
    W = np.zeros((N,N))
    # W[0,1] = W[1,0] = -2.0
    drives = [1]
    dt = 0.5
    net = Network(populations, W, drives, dt, history_len=int(40000.0/dt))
    net.run(int(15000/dt)) # runs for 15 seconds
    v = np.array(net.v_history)
    fr = net.firing_rate(v, v_half=-30, slope=4)
    plt.plot(fr[:, 0], color='b')
    # plt.plot(fr[:, 1], color='r')
    plt.show(block=True)




