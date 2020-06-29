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
        self.pop_names = list(populations.keys())
        self.N = len(self.populations)
        self.W = synaptic_weights
        self.W_neg = np.maximum(-self.W, 0)
        self.W_pos = np.maximum(self.W, 0)

        self.drives = drives
        self.dt = dt
        self.v = -100*np.random.rand(self.N) #np.ones(self.N) #
        self.m_ad = 0.4 + 0.0 * np.random.rand(self.N)

        self.input_cur = np.zeros(self.N)
        self.names = []
        self.C, self.g_ad, self.g_l, self.g_synE, self.g_synI, self.E_l, self.E_ad, self.E_synE, \
        self.E_synI, self.V_half, self.slope, self.K_ad, self.tau_ad = [np.zeros(self.N) for i in range(13)]
        self.v_history = deque(maxlen=self.history_len)
        self.m_ad_history = deque(maxlen=self.history_len)
        self.t = deque(maxlen=self.history_len)
        self.v_history.append(self.v)
        self.m_ad_history.append(self.m_ad)
        self.t.append(0)

        #load neural parameters into the internal variables: "self.C[i] = population[i].C"
        for i, (name, population) in enumerate(populations.items()):
            self.names.append(name)
            params_list = ["C", "g_ad", "g_l", "g_synE", "g_synI", "E_l",
                        "E_ad", "E_synE", "E_synI", "V_half", "slope", "K_ad", "tau_ad"]
            for p_name in params_list:
                exec(f'self.{p_name}[i] = population.{p_name}')

    def set_input_current(self, new_input_current):
        self.input_cur = deepcopy(new_input_current)
        return None

    def firing_rate(self, v, v_half, slope):
        return 1.0 / (1.0 + np.exp(-(v - v_half) / slope))

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


    def rhs_m_ad(self, v, m_ad):
        res = np.zeros_like(m_ad)
        c = self.g_ad != 0.0
        res[c] = (self.K_ad[c] * self.firing_rate(v[c], self.V_half[c], self.slope[c]) - m_ad[c]) / self.tau_ad[c]
        return res


    def rhs_v(self, v, m_ad):
        return (1.0 / self.C) * \
               (- self.I_adaptation(v, m_ad) - self.I_leakage(v)
                - self.I_SynE(v) - self.I_SynI(v) + self.input_cur)

    def step(self):
        #Runge-Kutta 4th order update
        k_v1 = self.dt * self.rhs_v(self.v, self.m_ad)
        k_m1 = self.dt * self.rhs_m_ad(self.v, self.m_ad)

        k_v2 = self.dt * self.rhs_v(self.v + k_v1 / 2, self.m_ad + k_m1 / 2)
        k_m2 = self.dt * self.rhs_m_ad(self.v + k_v1 / 2, self.m_ad + k_m1 / 2)

        k_v3 = self.dt * self.rhs_v(self.v + k_v2 / 2, self.m_ad + k_m2 / 2)
        k_m3 = self.dt * self.rhs_m_ad(self.v + k_v2 / 2, self.m_ad + k_m2 / 2)

        k_v4 = self.dt * self.rhs_v(self.v + k_v3, self.m_ad + k_m3)
        k_m4 = self.dt * self.rhs_m_ad(self.v + k_v3, self.m_ad + k_m3)


        new_v = self.v + 1.0 / 6.0 * (k_v1 + 2 * k_v2 + 2 * k_v3 + k_v4)
        new_m_ad = self.m_ad + 1.0 / 6.0 * (k_m1 + 2 * k_m2 + 2 * k_m3 + k_m4)


        self.v = new_v
        self.m_ad = new_m_ad
        return None

    def run(self, T_steps):
        for i in range(T_steps):
            self.step()
            self.v_history.append(deepcopy(self.v))
            self.m_ad_history.append(deepcopy(self.m_ad))
            self.t.append(self.t[-1] + self.dt)

    def plot(self):
        V_array = np.array(self.v_history).T
        t_array = np.array(self.t)
        fig, axes = plt.subplots(self.N, 1, figsize=(15, 10))
        if type(axes) != np.ndarray: axes = [axes]
        fr = self.firing_rate(V_array.T, self.V_half, self.slope).T
        for i in range(self.N): # we dont need inhibitor populations
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
        return fig, axes