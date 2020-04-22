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

    population_names = ['one', 'two', 'three']

    #create populations
    for name in population_names:
        exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")

    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    N = len(population_names)
    W = np.zeros((N,N))
    W[0,1] = W[1,0] = -1.0
    W[2, 1] = W[2, 0] = -2
    drives = np.array([[5, 5, 0]]).reshape(1, -1)
    dt = 0.25
    t_start_1 = 15000
    t_start_2 = 25000
    duration = 10
    t_stop = 40000
    net = Network(populations, W, drives, dt, history_len=int(40000.0/dt))
    net.v = np.array([-100, 100, -100])
    net.run(int(t_start_1/dt))
    # set an impuls input to "three" neuron
    inp = np.zeros(net.N)
    inp[2] = 200
    net.set_input_current(inp)
    net.run(int((duration) / dt))
    # run the network further
    net.set_input_current(np.zeros(net.N))
    net.run(int((t_start_2 - (t_start_1 + duration)) / dt))

    inp[2] = 200
    net.set_input_current(inp)
    net.run(int((duration) / dt))
    # run the network further
    net.set_input_current(np.zeros(net.N))
    net.run(int((t_stop - (t_start_2 + duration)) / dt))

    v = np.array(net.v_history)
    fr = net.firing_rate(v, v_half=-30, slope=4)
    plt.plot(net.t, fr[:, 0], color='b', label="one")
    plt.plot(net.t, fr[:, 1], color='r', label="two")
    plt.plot(net.t, fr[:, 2], color='k', label="three")
    plt.legend()
    plt.show(block=True)




