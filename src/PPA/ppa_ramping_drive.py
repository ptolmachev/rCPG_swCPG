import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy
from collections import deque
from src.PPA.simplified_model import *
from utils.gen_utils import get_project_root, create_dir_if_not_exist
from tqdm.auto import tqdm

default_neural_params = {
    'C' : 20,
    'g_ad' : 10.0,
    'g_l' : 2.8,
    'g_synE' : 10,
    'g_synI' : 60,
    'E_ad' : -85,
    'E_l' : -60,
    'E_synE' : 0,
    'E_synI' : -75,
    'V_half' : -30,
    'slope' : 4,
    'tau_ad' : 1000,
    'K_ad' : 0.9}
C = default_neural_params['C']
g_ad = default_neural_params['g_ad']
g_l = default_neural_params['g_l']
g_synE = default_neural_params['g_synE']
g_synI = default_neural_params['g_synI']
E_ad = default_neural_params['E_ad']
E_l = default_neural_params['E_l']
E_synE = default_neural_params['E_synE']
E_synI = default_neural_params['E_synI']
V_half = default_neural_params['V_half']
slope = default_neural_params['slope']
tau_ad = default_neural_params['tau_ad']
K_ad = default_neural_params['K_ad']

def firing_rate(v, v_half, slope):
    return 1.0 / (1.0 + np.exp(-(v - v_half) / slope))


def firing_rate_inv(s, v_half, slope):
    return -np.log(1.0 / s - 1.0) * slope + v_half


def I_leakage(v):
    res = g_l * (v - E_l)
    return res


def I_adaptation(v, m_ad):
    res = g_ad * m_ad * (v - E_ad)
    return res


def I_adaptation_inf(v):
    res = g_ad * K_ad * firing_rate(v, V_half, slope) * (v - E_ad)
    return res


def I_tonicE(v, drive):
    I = g_synE * (v - E_synE) * drive
    return I


def I_tonicI(v, w, fr):
    return g_synI * (v - E_synI) * w * fr


def rhs_v(v, m_ad, drive, w, fr):
    return (1 / C) * (-I_leakage(v) - I_adaptation(v, m_ad) - I_tonicE(v, drive) - I_tonicI(v, w, fr))


def rhs_m_ad(v, m_ad):
    res = (K_ad * firing_rate(v, V_half, slope) - m_ad) / tau_ad
    return res


def v_nullcline(v, drive, w, fr):
    m = - (I_leakage(v) + I_tonicE(v, drive) + I_tonicI(v, w, fr)) / (g_ad * (v - E_ad))
    return m


def m_nullcline(v):
    m = K_ad * firing_rate(v, V_half, slope)
    return m


def plot_nullclines_2(drives, ws, frs, x_lims, y_lims):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    v1 = np.linspace(*x_lims[0], 1200)
    x1 = np.linspace(*x_lims[0], 10)
    y1 = np.linspace(*y_lims[0], 10)
    X1, Y1 = np.meshgrid(x1, y1)
    dv1dt = rhs_v(X1, Y1, drives[0], ws[1],frs[1])
    dm1dt = rhs_m_ad(X1, Y1)
    axes[0].plot(v1, v_nullcline(v1, drives[0], ws[1], frs[1]), color='r')
    axes[0].quiver(X1, Y1, dv1dt, dm1dt)
    axes[0].plot(v1, m_nullcline(v1), color='green')
    axes[0].set_xlim(x_lims[0])
    axes[0].set_ylim(y_lims[0])

    v2 = np.linspace(*x_lims[1], 1200)
    x2 = np.linspace(*x_lims[1], 10)
    y2 = np.linspace(*y_lims[1], 10)
    X2, Y2 = np.meshgrid(x2, y2)
    dv2dt = rhs_v(X2, Y2, drives[1], ws[0],frs[0])
    dm2dt = rhs_m_ad(X2, Y2)
    axes[1].plot(v2, v_nullcline(v2, drives[1], ws[0], frs[0]), color='r')
    axes[1].quiver(X2, Y2, dv2dt, dm2dt)
    axes[1].plot(v2, m_nullcline(v2), color='green')
    axes[1].set_xlim(x_lims[1])
    axes[1].set_ylim(y_lims[1])

    return fig, axes

if __name__ == '__main__':
    #plotting PPA
    img_folder = f"{get_project_root()}/img"
    save_imgs_to = f'{img_folder}/ppa/ppa_ramping_drive'
    create_dir_if_not_exist(save_imgs_to)

    population_names = ["One", "Two"]
    for name in population_names:
        exec(f"{name} = NeuralPopulation(\'{name}\', default_neural_params)")
    One.tau_ad = 1000.0
    Two.tau_ad = 1000.0
    # populations dictionary
    populations = dict()
    for name in population_names:
        populations[name] = eval(name)

    W = np.zeros((2, 2))
    W[0, 1] = -0.35
    W[1, 0] = -0.47
    drives = np.array([[0.4, 0.4]])
    dt = 0.5
    net = Network(populations, W, drives, dt, history_len=int(200000 / dt))

    T_steps = int(50000 / dt)
    for i in range(T_steps):
        net.drives = np.array([0.15 * np.ones(2) + 0.7 * (i/T_steps)])
        net.step()
        net.v_history.append(deepcopy(net.v))
        net.m_ad_history.append(deepcopy(net.m_ad))
        net.t.append(net.t[-1] + net.dt)

    transients = 10000
    v = np.array(net.v_history)[transients:]
    m = np.array(net.m_ad_history)[transients:]
    t = np.array(net.t)[transients:]

    fig, ax = plt.subplots(5, 1, figsize=(14, 5))
    ax[0].plot(t, v[:, 0], color='r')
    ax[1].plot(t, v[:, 1], color='r')
    ax[2].plot(t, m[:, 0], color='b')
    ax[3].plot(t, m[:, 1], color='b')
    ax[4].plot(t,  0.15 + 0.7 * (np.arange(T_steps)[transients-1:]/T_steps), color='b')
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig(f'{save_imgs_to}/traces.png')
    # plt.show()

    # looping over the cycle:
    # w is unchanged, fr changes: which leads to different position of the nullclines.
    ws = np.array([np.abs(W[0, 1]), np.abs(W[1, 0])])
    drives = drives.squeeze()
    x_lims = np.array([[-80, -10], [-80, -10]])
    y_lims = np.array([[0, 1], [0, 1]])

    m_min = np.min(m, axis=0)
    m_avg = np.mean(m, axis=0)
    m_max = np.max(m, axis=0)

    v_min = np.min(v, axis=0)
    v_avg = np.mean(v, axis=0)
    v_max = np.max(v, axis=0)

    y_lims = np.array([m_avg + (m_min - m_avg) * 1.5, m_avg + (m_max - m_avg) * 1.5]).T
    x_lims = np.array([v_avg + (v_min - v_avg) * 1.5, v_avg + (v_max - v_avg) * 1.5]).T

    c = 0
    for i in tqdm(range(T_steps)):
        if i % 50 == 0:
            v1_c = v[i, 0]
            v2_c = v[i, 1]
            m1_c = m[i, 0]
            m2_c = m[i, 1]
            fr_1 = firing_rate(v1_c, V_half, slope)
            fr_2 = firing_rate(v2_c, V_half, slope)
            frs = np.array([fr_1, fr_2])
            drives = 0.15 * np.ones(2) + 0.7 * ( (i+transients-1) / T_steps)
            fig, axes = plot_nullclines_2(drives, ws, frs, x_lims, y_lims)

            # axes[0].plot(v[:, 0], m[:, 0], color='orange')
            axes[0].plot(v[i, 0], m[i, 0], 'bo', markersize=5)
            for j in range(1000):
                axes[0].plot(v[np.maximum(0, i - j), 0], m[np.maximum(0, i - j), 0], 'bo', alpha=((1000 - j) / 1000),
                             markersize=0.5)

            # axes[1].plot(v[:, 1], m[:, 1], color='orange')
            axes[1].plot(v[i, 1], m[i, 1], 'bo', markersize=5)
            for j in range(1000):
                axes[1].plot(v[np.maximum(0, i - j), 1], m[np.maximum(0, i - j), 1], 'bo', alpha=((1000 - j) / 1000),
                             markersize=0.5)

            axes[0].set_xlabel("v 1, mV")
            axes[0].set_ylabel("m 1, units")
            axes[1].set_xlabel("v 2, mV")
            axes[1].set_ylabel("m 2, units")
            fig.suptitle(f"Drive = {np.round(drives[0],3)}")

            plt.savefig(f'{save_imgs_to}/{str.zfill(str(c), 5)}.png')
            c += 1
            plt.close()