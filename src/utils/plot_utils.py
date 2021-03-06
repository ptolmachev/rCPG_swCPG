import pickle

import numpy as np
from matplotlib import pyplot as plt
from src.utils.sp_utils import get_onsets_and_ends, get_VNA_ends
from utils.gen_utils import get_project_root
from scipy.signal import savgol_filter
from scipy.signal import periodogram
from numpy.polynomial.polynomial import Polynomial

def plot_power_spectrum(rec, fs, fr_low):
    img_path = str(get_project_root()) + "/img"
    f, Pxx_den = periodogram(rec, fs)
    fig = plt.figure(figsize=(20,5))
    inds = np.where(f > fr_low)
    plt.plot(f[inds], Pxx_den[inds])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show(block=True)
    fig.savefig(img_path + f"/periodogram.png")
    return None

def nice_error_bar(x,y,error, title, xlabel,ylabel, save_to = None):
    img_path = str(get_project_root()) + "/img"
    fig = plt.figure(figsize = (40,20))
    plt.errorbar(x,y, yerr = error , color = 'red',
             ecolor = 'gray', capsize = 3,linestyle = 'dashed', linewidth = 3, alpha = 0.8)
    plt.title(title, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.ylim(min(y-error)-0.2*abs(min(y-error)),1.2*max(y+error))
    plt.grid(True)
    plt.show()
    if save_to is not None:
        fig.savefig(img_path + '/' + save_to)

def nice_error_bar_scatter(x,y,error, title, xlabel,ylabel, save_to = None):
    img_path = str(get_project_root()) + "/img"
    fig = plt.figure(figsize = (40,20))
    plt.errorbar(x, y, yerr = error , color = 'red', ls='', marker='o',
             ecolor = 'gray', capsize = 3, linewidth = 3, alpha = 0.8)
    plt.title(title, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.ylim(min(y-error)-0.2*abs(min(y-error)),1.2*max(y+error))
    plt.grid(True)
    plt.show()
    if save_to is not None:
        fig.savefig(img_path + '/' + save_to)

def scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, phi_insp=None, deg=None):
    fig = plt.figure()
    plt.title(title)
    plt.scatter(x, y)
    if fit_poly:
        p = Polynomial.fit(x, y, deg=deg)
        plt.plot(np.sort(x), p(np.sort(x)), color='r', linewidth=3)
    if not phi_insp is None:
        plt.axvline(phi_insp, color='k', linestyle='--')
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not y_lim is None:
        plt.ylim(y_lim)
    if not x_lim is None:
        plt.xlim(x_lim)
    # plt.show(block=True)
    # plt.savefig(path_save_to)
    return fig

def plot_chunk(data_chunk, y_lim):
        PNA = data_chunk['PNA']
        HNA = data_chunk['HNA']
        VNA = data_chunk['VNA']
        stim_start = data_chunk["stim_start"]
        stim_end = data_chunk["stim_end"]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 9))
        signals = [PNA, HNA, VNA]
        signals = [savgol_filter(s, 5, 3) for s in signals]
        labels = ["PNA", "HNA", "VNA"]
        starts, ends = get_onsets_and_ends(PNA, model='l2', pen=1000, min_len=60)
        ends_VNA = get_VNA_ends(VNA, starts, ends)
        for i in range(3):
            ax = eval(f"ax{i+1}")
            ax.plot(signals[i], 'k', linewidth=2)
            for j in range(len(starts)):
                ax.axvline(starts[j], color="r")
            for j in range(len(ends)):
                ax.axvline(ends[j], color="b")
            if i == 2:
                # for j in range(len(starts_VNA)):
                #     ax.axvline(starts_VNA[j], color="magenta")
                for j in range(len(ends_VNA)):
                    ax.axvline(ends_VNA[j], color="orange")

            ax.axvline(stim_start, color="r", linewidth=2, linestyle = "-")
            ax.axvline(stim_end, color="r", linewidth=2, linestyle="-")
            ax.axvspan(stim_start, stim_end, color="r", alpha=0.1)
            ax.grid(True)
            if i != 2:
                ax.set_xticklabels([])
            ax.set_ylabel(labels[i], fontdict={"fontsize" : 16})
            ax.set_ylim(y_lim)
        # plt.show(block = True)
        # fig.savefig(save_to)
        # plt.close()
        plt.subplots_adjust(wspace=0, hspace=0)
        return fig

def features_plots(file_load, dir_save_to, modifier, plot_params):
    fit_poly = plot_params["fit_poly"]
    y_lim_T0 = plot_params["y_lim_T0"]
    y_lim_T1 = plot_params["y_lim_T1"]
    y_lim_Ti1 = plot_params["y_lim_Ti1"]
    y_lim_T1divT0 = plot_params["y_lim_T1divT0"]

    data_ = pickle.load(open(file_load,'rb'))['data']
    data_ = np.array(data_)
    # data = get_rid_of_outliers(data)
    Phi = data_[:, 0]
    Ti_0 = data_[:, 1]
    T0 = data_[:, 2]
    T1 = data_[:, 3]
    Theta = data_[:, 4]
    Ti_1 = data_[:, 5]
    Ti_2 = data_[:, 6]
    phase = (Phi/np.nanmean(T0))
    cophase = (Theta/np.nanmean(T0))
    phi_insp = np.nanmean(Ti_0)/np.nanmean(T0)

    x_label = "Phase"
    x = phase
    x_lim = [0, 1.2]

    title = "Phase-Cophase"
    y_label = "Cophase"
    y = cophase
    y_lim = [0, 1.1]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    fig = scatterplot(x, y, x_label, y_label, x_lim, y_lim , title, fit_poly, phi_insp=phi_insp, deg=6)
    fig.savefig(path_save_to)

    title = "T1 T0 ratio"
    y_label = "T1/T0"
    y = T1/T0
    y_lim = y_lim_T1divT0
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    fig = scatterplot(x, y, x_label, y_label, x_lim, y_lim , title, fit_poly, phi_insp=phi_insp, deg=6)
    fig.savefig(path_save_to)

    title = "T0"
    y_label = "T0, ms"
    y = T0
    y_lim = y_lim_T0
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    fig = scatterplot(x, y, x_label, y_label, x_lim, y_lim , title, fit_poly, phi_insp=phi_insp, deg=0)
    fig.savefig(path_save_to)

    title = "T1"
    y_label = "T1, ms"
    y = T1
    y_lim = y_lim_T1
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    fig = scatterplot(x, y, x_label, y_label, x_lim, y_lim , title, fit_poly, phi_insp=phi_insp, deg=6)
    fig.savefig(path_save_to)

    title = "Ti 1"
    y_label = "Ti 1, ms"
    y = Ti_1
    y_lim = y_lim_Ti1
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    fig = scatterplot(x, y, x_label, y_label, x_lim, y_lim , title, fit_poly, phi_insp=phi_insp, deg=0)
    fig.savefig(path_save_to)
    return None

def plot_num_exp_traces(signals):
    N = signals.shape[0]
    names = ['PreI', 'EarlyI', "PostI", "AugE","RampI", "Relay", "Sw1","Sw2","Sw3", "KF_t", "KF_p", "KF_relay", "HN",  "PN",  "VN", "KF_inh", "NTS_inh"]  # 16
    fig, axes = plt.subplots(N - 2, 1, figsize=(25, 15))
    if type(axes) != np.ndarray: axes = [axes]
    for i in range(N - 2):  # we dont need inhibitor populations
        if i == 0: axes[i].set_title('Firing Rates', fontdict={"size": 25})
        axes[i].plot(signals[i, :], 'k', linewidth=3, label=str(names[i]), alpha=0.9)
        axes[i].legend(loc=1, fontsize=25)
        axes[i].set_ylim([-0.0, 1.0])
        axes[i].set_yticks([])
        axes[i].set_yticklabels([])
        if i != len(axes) - 1:
            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
        axes[i].set_xlabel('t, ms', fontdict={"size": 25})
    plt.subplots_adjust(wspace=0.01, hspace=0)
    return fig



# if __name__ == '__main__':


    # # PLOT PSD
    # data_folder = '../../data/sln_prc'
    # folder_save_to = '../../data/sln_prc_filtered'
    # folders = get_folders(data_folder, "prc")
    # folder = folders[1]
    # suffix = 'CH10'
    # signals = load(f'{data_folder}/{folder}/100_{suffix}.continuous', dtype=float)["data"]
    # plot_power_spectrum(signals, 30000)




