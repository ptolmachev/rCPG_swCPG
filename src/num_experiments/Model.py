import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from collections import deque
from num_experiments.params_gen import generate_params
from src.utils.gen_utils import get_postfix


class NeuralPopulation():
    def __init__(self, name, params):
        self.name = name
        #sets all the internal variables
        for p_name in list(params.keys()):
            exec(f"self.{p_name} = params[\"{p_name}\"]")

class Network():
    def __init__(self, populations, synaptic_weights, drives, dt, history_len):
        # should be a dictionary
        self.history_len = history_len
        self.populations = populations
        self.N = len(self.populations)
        self.W = synaptic_weights
        self.W_neg = np.maximum(-self.W, 0)
        self.W_pos = np.maximum(self.W, 0)

        self.drives = drives
        self.dt = dt
        self.v = -100*np.random.rand(self.N) #np.ones(self.N) #
        self.h_NaP = 0.4 + 0.0 * np.random.rand(self.N)
        self.m_ad = 0.4 + 0.0 * np.random.rand(self.N)

        self.input_cur = np.zeros(self.N)
        self.names = []
        self.C, self.g_NaP, self.g_K, self.g_ad, self.g_l, self.g_synE, self.g_synI, self.E_Na, self.E_K, self.E_l,\
        self.E_ad, self.E_synE, self.E_synI, self.V_half, self.slope, self.K_ad, self.tau_ad, self.tau_NaP_max = [np.zeros(self.N) for i in range(18)]
        self.v_history = deque(maxlen=self.history_len)
        self.t = deque(maxlen=self.history_len)
        self.v_history.append(self.v)
        self.t.append(0)

        #load neural parameters into the internal variables: "self.C[i] = population[i].C"
        for i, (name, population) in enumerate(populations.items()):
            self.names.append(name)
            params_list = ["C", "g_NaP", "g_K", "g_ad", "g_l", "g_synE", "g_synI", "E_Na", "E_K", "E_l",
                        "E_ad", "E_synE", "E_synI", "V_half", "slope", "K_ad", "tau_ad", "tau_NaP_max"]
            for p_name in params_list:
                exec(f'self.{p_name}[i] = population.{p_name}')

    def set_input_current(self, new_input_current):
        self.input_cur = deepcopy(new_input_current)
        return None

    def firing_rate(self, v, v_half, slope):
        return 1.0 / (1.0 + np.exp(-(v - v_half) / slope))

    def m_NaP(self, v):
        return 1.0 / (1.0 + np.exp(-(v + 40.0) / 6.0))

    def m_K(self, v):
        return 1.0 / (1.0 + np.exp(-(v + 29.0) / 4.0))

    def h_NaP_inf(self, v):
        return 1.0 / (1.0 + np.exp((v + 48.0) / 6.0))

    def tau_NaP(self, v, tau_NaP_max):
        return tau_NaP_max / np.cosh((v + 48.0) / 12.0)

    def I_NaP(self, v, h_NaP):
        res = np.zeros_like(v)
        c = self.g_NaP != 0.0
        res[c] = self.g_NaP[c] * self.m_NaP(v[c]) * h_NaP[c] * (v[c] - self.E_Na[c])
        return res

    def I_K(self, v):
        res = np.zeros_like(v)
        c = self.g_K != 0.0
        res[c] = self.g_K[c] * ((self.m_K(v[c])) ** 4) * (v[c] - self.E_K[c])
        return res

    def I_leakage(self, v):
        res = np.zeros_like(v)
        c = self.g_l != 0.0
        res[c] = self.g_l[c] * (v[c] - self.E_l[c])
        return res

    def I_adaptation(self, v, m_ad):
        res = np.zeros_like(v)
        c = self.g_ad != 0.0
        res[c] = self.g_ad[c] * m_ad[c] * (v[c] - self.E_ad[c])
        return res

    def I_SynE(self, v):
        tonic_drives_all = np.sum(self.drives, axis=0)
        I_tonicE = self.g_synE * (v - self.E_synE) * tonic_drives_all
        I_synE = I_tonicE + self.g_synE * (v - self.E_synE) * \
                 (self.firing_rate(v, self.V_half, self.slope).reshape(1, self.N) @  self.W_pos).flatten()
        return I_synE

    def I_SynI(self, v):
        I_synI = self.g_synI * (v - self.E_synI) * \
                 (self.firing_rate(v, self.V_half, self.slope).reshape(1, self.N) @  self.W_neg).flatten()
        return I_synI

    def rhs_h_NaP(self, v, h_NaP):
        res = np.zeros_like(h_NaP)
        c = self.g_NaP != 0.0
        res[c] = (self.h_NaP_inf(v[c]) - h_NaP[c]) / self.tau_NaP(v[c], self.tau_NaP_max[c])
        return res

    def rhs_m_ad(self, v, m_ad):
        res = np.zeros_like(m_ad)
        c = self.g_ad != 0.0
        res[c] = (self.K_ad[c] * self.firing_rate(v[c], self.V_half[c], self.slope[c]) - m_ad[c]) / self.tau_ad[c]
        return res


    def rhs_v(self, v, m_ad, h_NaP):
        return (1.0 / self.C) * \
               (- self.I_NaP(v, h_NaP) - self.I_adaptation(v, m_ad) - self.I_leakage(v)
                - self.I_SynE(v) - self.I_SynI(v) + self.input_cur)

    def step(self):
        #Runge-Kutta 4th order update
        k_v1 = self.dt * self.rhs_v(self.v, self.m_ad, self.h_NaP)
        k_m1 = self.dt * self.rhs_m_ad(self.v, self.m_ad)
        k_h1 = self.dt * self.rhs_h_NaP(self.v, self.h_NaP)

        k_v2 = self.dt * self.rhs_v(self.v + k_v1 / 2, self.m_ad + k_m1 / 2, self.h_NaP + k_h1 / 2)
        k_m2 = self.dt * self.rhs_m_ad(self.v + k_v1 / 2, self.m_ad + k_m1 / 2)
        k_h2 = self.dt * self.rhs_h_NaP(self.v + k_v1 / 2, self.h_NaP + k_h1 / 2)

        k_v3 = self.dt * self.rhs_v(self.v + k_v2 / 2, self.m_ad + k_m2 / 2, self.h_NaP + k_h2 / 2)
        k_m3 = self.dt * self.rhs_m_ad(self.v + k_v2 / 2, self.m_ad + k_m2 / 2)
        k_h3 = self.dt * self.rhs_h_NaP(self.v + k_v2 / 2, self.h_NaP + k_h2 / 2)

        k_v4 = self.dt * self.rhs_v(self.v + k_v3, self.m_ad + k_m3, self.h_NaP + k_h3)
        k_m4 = self.dt * self.rhs_m_ad(self.v + k_v3, self.m_ad + k_m3)
        k_h4 = self.dt * self.rhs_h_NaP(self.v + k_v3, self.h_NaP + k_h3)

        new_v = self.v + 1.0 / 6.0 * (k_v1 + 2 * k_v2 + 2 * k_v3 + k_v4)
        new_m_ad = self.m_ad + 1.0 / 6.0 * (k_m1 + 2 * k_m2 + 2 * k_m3 + k_m4)
        new_h_NaP = self.h_NaP + 1.0 / 6.0 * (k_h1 + 2 * k_h2 + 2 * k_h3 + k_h4)

        self.v = new_v
        self.m_ad = new_m_ad
        self.h_NaP = new_h_NaP
        return None

    def run(self, T_steps):
        for i in range(T_steps):
            self.step()
            self.v_history.append(deepcopy(self.v))
            self.t.append(self.t[-1] + self.dt)

    def plot(self, show, save_to):
        V_array = np.array(self.v_history).T
        t_array = np.array(self.t)
        fig, axes = plt.subplots(self.N - 2, 1, figsize=(25, 15))
        if type(axes) != np.ndarray: axes = [axes]
        fr = self.firing_rate(V_array.T, self.V_half, self.slope).T
        for i in range(self.N - 2): # we dont need inhibitor populations
            if i == 0: axes[i].set_title('Firing Rates', fontdict={"size" : 25})
            axes[i].plot(t_array, fr[i, :], 'k', linewidth=3, label=str(self.names[i]), alpha=0.9)
            axes[i].legend(loc = 1, fontsize=25)
            axes[i].set_ylim([-0.0, 1.0])
            axes[i].set_yticks([])
            axes[i].set_yticklabels([])
            if i != len(axes) - 1:
                axes[i].set_xticks([])
                axes[i].set_xticklabels([])
            axes[i].set_xlabel('t, ms', fontdict={"size" : 25})
        plt.subplots_adjust(wspace=0.01, hspace=0)
        fig.savefig(save_to)
        if show:
            plt.show()
        plt.close()
        return None


if __name__ == '__main__':
    default_neural_params = {
    'C' : 20,
    'g_NaP' : 0.0,
    'g_K' : 5.0,
    'g_ad' : 10.0,
    'g_l' : 2.8,
    'g_synE' : 10,
    'g_synI' : 60,
    'g_synE_slow' : 0,
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

    population_names = ['PreI',  # 0
                        'EarlyI',  # 1
                        "PostI",  # 2
                        "AugE",  # 3
                        "RampI",  # 4
                        "Relay",  # 5
                        "Sw1",  # 6
                        "Sw2",  # 7
                        "Sw3",  # 8
                        "KF_t",  # 9
                        "KF_p",  # 10
                        "KF_relay",  # 11
                        "HN",  # 12
                        "PN",  # 13
                        "VN",  # 14
                        "KF_inh",  # 15
                        "NTS_inh", #16
                        "SI"]  # 17

    #create populations
    for name in population_names:
        exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")

    #modifications:
    PreI.g_NaP = 5.0
    PreI.g_ad = HN.g_ad = PN.g_ad = VN.g_ad = SI.g_ad =  0.0
    HN.g_NaP = PN.g_NaP = VN.g_NaP = SI.g_NaP  = 0.0
    Relay.tau_ad = 15000.0
    PostI.tau_ad = 10000.0
    Sw1.tau_ad = 1000.0
    Sw2.tau_ad = 1000.0

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
    dt = 0.5
    net = Network(populations, W, drives, dt, history_len=int(40000/dt))
    # get rid of all transients
    net.run(int(15000/dt)) # runs for 15 seconds
    # run for 15 more seconds
    net.run(int(15000/dt))
    #set input to Relay neurons
    inp = np.zeros(net.N)
    inp[17] = 370
    net.set_input_current(inp)
    # run for 10 more seconds
    net.run(int(10000/dt))
    net.set_input_current(np.zeros(net.N))
    # run for 15 more seconds
    net.run(int(15000/dt))
    # create_dir_if_not_exist("../img/")
    # net.plot(show = True, save_to = f"../img/Model_10_02_2020/{get_postfix(inh_NTS, inh_KF)}.png")
    v = np.array(net.v_history)
    fr = net.firing_rate(v, v_half=-30, slope=4)
    plt.plot(fr[:, 5], color='b')
    plt.plot(fr[:, 17], color='r')
    plt.plot(fr[:, 2], color='green')
    plt.show(block=True)




