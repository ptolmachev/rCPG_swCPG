import pickle
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from matplotlib import rc
from sp_utils import get_onsets_and_ends, find_relevant_peaks
from utils import create_dir_if_not_exist, get_files, get_folders
from scipy.signal import savgol_filter
from copy import deepcopy
# rc('text', usetex=True)
from scipy.signal import periodogram

def plot_power_spectrum(rec, fs):
    f, Pxx_den = periodogram(rec, fs)
    fig = plt.figure(figsize=(20,5))
    inds = np.where(f > 300)
    plt.plot(f[inds], Pxx_den[inds])
    # plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show(block=True)
    fig.savefig(f"../../img/periodogram.png")
    return None

def nice_error_bar(x,y,error, title, xlabel,ylabel, save_to = None):
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
        fig.savefig("../img/" + save_to)


def nice_error_bar_scatter(x,y,error, title, xlabel,ylabel, save_to = None):

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
        fig.savefig("../img/" + save_to)

def scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=None, deg=None):
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
    plt.savefig(path_save_to)
    return None

def features_plots(file_load, dir_save_to, modifier, fit_poly):
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
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=6)

    title = "T1 div T0"
    y_label = "T1/T0"
    y = T1/T0
    y_lim = [0, 1.75]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=6)

    title = "T0"
    y_label = "T0, ms"
    y = T0
    y_lim = [0, 5000]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=0)

    title = "T1"
    y_label = "T1, ms"
    y = T1
    y_lim = [0,5000]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=6)

    title = "Ti 1"
    y_label = "Ti 1, ms"
    y = Ti_1
    y_lim = [0,1500]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim , title, fit_poly, path_save_to, phi_insp=phi_insp, deg=0)
    return None

def clarifying_plot(chunk, save_to):
    PNA = chunk['signal']
    s = int(0.4 * len(PNA))
    e = int(0.8 * len(PNA))
    stim = chunk['stim'] - s
    PNA = (PNA[s:e])
    threshold = 7.5
    min_len = 50
    insp_begins, insp_ends = get_inspiration_onsets_and_ends(PNA, threshold, min_len)
    ts1, ts2, ts3, ts4, te1, te2, te3, te4 = get_onsets_and_ends(insp_begins, insp_ends, stim)
    PNA = (PNA - np.min(PNA)) / (np.max(PNA) - np.min(PNA))

    class DoubleArrow():
        def __init__(self, pos1, pos2, level, margin):
            plt.arrow(pos1 + margin, level, pos2 - pos1 - 2 * margin, 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k')
            plt.arrow(pos2, level, pos1 - pos2 + 2 * margin , 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k', head_starts_at_zero = True)

    class ConvergingArrows():
        def __init__(self, pos1, pos2, level, margin):
            plt.arrow(pos1+10 - 10*margin, level, 8*margin, 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k')
            plt.arrow(pos2 + 10*margin, level, -8* margin , 0.0, shape='full',
                      length_includes_head =True, head_width=0.03,
                      head_length=20, fc='k', ec='k', head_starts_at_zero = True)


    fig = plt.figure(figsize=(20, 6))
    plt.plot(PNA, linewidth=2, color='k')
    margin = 5
    height0 = 0.9
    height1 = 1.05
    height2 = 0.00
    height3 = -0.05
    stim_duration = 75
    ts1 = ts1-40
    ts2 = ts2 - 20
    ConvergingArrows(stim, stim+stim_duration, height0, margin)  # Stim
    DoubleArrow(ts1, ts2, height1, margin) # T0
    DoubleArrow(ts2, ts3, height1, margin)  # T1
    DoubleArrow(ts1, te1, height2, margin)  # Ti_0
    DoubleArrow(ts3, te3, height2, margin)  # Ti_1
    DoubleArrow(ts4, te4, height2, margin)  # Ti_2
    DoubleArrow(ts2, stim, height3, margin)  # Phi
    DoubleArrow(stim+stim_duration, ts3, height3, margin)  # Theta

    plt.axvline(stim, color='r', linestyle='--')
    plt.axvline(stim + stim_duration, color='r', linestyle='--')
    plt.axvspan(stim, stim + stim_duration, color='r', alpha=0.3)
    plt.axvline(ts1, color='b', linestyle='--')
    plt.axvline(ts2, color='b', linestyle='--')
    plt.axvline(ts3, color='b', linestyle='--')
    plt.axvline(ts4, color='b', linestyle='--')
    plt.axvline(te1, color='b', linestyle='--')
    plt.axvline(te2, color='b', linestyle='--')
    plt.axvline(te3, color='b', linestyle='--')
    plt.axvline(te4, color='b', linestyle='--')

    plt.title("Phrenic Nerve Activity", fontsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.1, 1.1])
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    plt.axis('off')
    plt.savefig(f"{save_to}")
    plt.close()
    return None

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
        # starts_VNA, ends_VNA = get_onsets_and_ends(VNA, model='l2', pen=100, min_len=60)
        for i in range(3):
            ax = eval(f"ax{i+1}")
            ax.plot(signals[i], 'k', linewidth=2)
            for j in range(len(starts)):
                ax.axvline(starts[j], color="r")
            for j in range(len(ends)):
                ax.axvline(ends[j], color="b")
            # if i != 2:
            #     for j in range(len(starts)):
            #         ax.axvline(starts[j], color="r")
            #     for j in range(len(ends)):
            #         ax.axvline(ends[j], color="b")
            # else:
            #     for j in range(len(starts_VNA)):
            #         ax.axvline(starts_VNA[j], color="r")
            #     for j in range(len(ends_VNA)):
            #         ax.axvline(ends_VNA[j], color="b")

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

def plot_num_exp_traces(signals, save_to):
    N = signals.shape[0]
    names = ['PreI', 'EarlyI', "PostI","AugE","RampI", "Relay", "Sw1","Sw2","Sw3", "KF_t", "KF_p", "KF_relay", "HN",  "PN",  "VN", "KF_inh", "NTS_inh"]  # 16
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
    fig.savefig(save_to)
    plt.close()
    return None

def combining_data(timestamp, inds):
    data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                  '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
    phase = np.array([])
    cophase = np.array([])
    T0_all = np.array([])
    T1_all = np.array([])
    for i in inds:
        num_rec = i
        file_load = f'../../data/parameters_prc_{timestamp}_{data_files[num_rec]}.pkl'
        dir_save_to = '../../img/experiments'
        data_ = pickle.load(open(file_load, 'rb'))['data']
        data_ = np.array(data_)
        Phi = data_[:, 0]
        T0 = data_[:, 2]
        T1 = data_[:, 3]
        Theta = data_[:, 4]
        phase = np.hstack([phase, (Phi / np.mean(T0))])
        cophase = np.hstack([cophase, (Theta / np.mean(T0))])
        T0_all = np.hstack([T0_all, T0])
        T1_all = np.hstack([T1_all, T1])

    fig1 = plt.figure()
    plt.title("Phase-Cophase")
    y = cophase
    p = Polynomial.fit(phase, y,deg=6)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.xlabel("Phase")
    plt.ylabel("Cophase")
    plt.savefig(f"{dir_save_to}/pulled_phase_cophase.png")

    fig2 = plt.figure()
    plt.title("T1/T0")
    y = T1_all/T0_all
    p = Polynomial.fit(phase, y,deg=6)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.ylim([0, 1.1 * np.max(y - np.mean(y)) + np.mean(y)])
    plt.xlabel("Phase")
    plt.ylabel("T1/T0 ratio")
    plt.savefig(f"{dir_save_to}/pulled_period_change.png")
    return None


if __name__ == '__main__':
    ### plotting final experimental data
    # timestamp = "21032020"
    # for i in range(4):
    #     num_rec = i
    #     data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
    #                   '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
    #     file_load = f'../data/parameters_prc_{timestamp}_{data_files[num_rec]}.pkl'
    #     dir_save_to = '../img/experiments'
    #     modifier = num_rec
    #     features_plots(file_load, dir_save_to, modifier, fit_poly=True)

    ### plotting numerical experimental traces
    # params = {}
    # amps = [150, 250, 350]
    # stim_duration = 500
    # data_path = "../data"
    # img_path = "../img"
    # for amp in amps:
    #     params["amp"] = amp
    #     folder_signals = f"num_exp_runs/num_exp_short_stim_{amp}_{stim_duration}"
    #     folder_save_img_to = f"traces"
    #     create_dir_if_not_exist(img_path + "/" + f"num_experiments/short_stim/short_stim_{amp}_{stim_duration}", folder_save_img_to)
    #     files = get_files(root_folder=f"../data/num_exp_runs/num_exp_short_stim_{amp}_{stim_duration}", pattern=".pkl")
    #     for file in files:
    #         data = pickle.load(open(data_path +"/" +folder_signals + "/" + file, "rb+"))
    #         save_to = ""
    #         signals = data['signals']
    #         name = file.split(".pkl")[0] + ".png"
    #         plot_num_exp_traces(signals, img_path + "/" + f"num_experiments/short_stim/short_stim_{amp}_{stim_duration}/traces/{name}")

    # plotting final numerical experimental data
    # amps = [100,200,300,400,500]
    # amps = [150, 250, 350]
    # stim_duration = 500
    # for amp in amps:
    #     file_load = f'../data/num_exp_results/short_stim/info_var_phase_{amp}_{stim_duration}.pkl'
    #     dir_save_to = f'../img/num_experiments/short_stim/short_stim_{amp}_{stim_duration}'
    #     modifier = f"{amp}_{stim_duration}"
    #     create_dir_if_not_exist(f'../img/num_experiments/short_stim', f"short_stim_{amp}_{stim_duration}")
    #     features_plots(file_load, dir_save_to, modifier, fit_poly=True)

    ### PLOTTING CHUNKS
    # num_rec = 1 - nice chunk
    # chunk_num = 19
    data_folders = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                    '2019-09-05_12-26-14_prc']
    for i in range(len(data_folders)):
        file_load = f'../data/sln_prc_chunked/{data_folders[i]}/chunked.pkl'
        data = pickle.load(open(file_load, 'rb+'))
        list_chunks = list(data.keys())
        for chunk_num in list_chunks:
            data_chunk = data[chunk_num]
            fig = plot_chunk(data_chunk, y_lim = [1, 18])
            fig.savefig(f"../img/experiments/traces/short_stim/{i}_{chunk_num}")
            plt.close(fig)

    # ### PLOTTING LONG STIM DATA
    # data_folders = get_folders(f"../data/sln_prc_filtered/", "_t")
    # for folder in (data_folders):
    #     # folder = deepcopy(data_folders[i])
    #     print(folder)
    #     HNA = pickle.load(open(f'../data/sln_prc_filtered/{folder}/100_CH5_processed.pkl', 'rb+'))
    #     PNA = pickle.load(open(f'../data/sln_prc_filtered/{folder}/100_CH10_processed.pkl', 'rb+'))
    #     VNA = pickle.load(open(f'../data/sln_prc_filtered/{folder}/100_CH15_processed.pkl', 'rb+'))
    #     data = {}
    #     stim_duration = int(PNA['stim_end'] - PNA['stim_start'])
    #     signal_start = int(np.maximum(0, PNA['stim_start'] - stim_duration))
    #     signal_end =  int(np.minimum(PNA['stim_end'] + stim_duration, len(PNA['signal']) - 1))
    #     data["HNA"] = deepcopy(HNA['signal'][signal_start:signal_end])
    #     data["PNA"] = deepcopy(PNA['signal'][signal_start:signal_end])
    #     data["VNA"] = deepcopy(VNA['signal'][signal_start:signal_end])
    #     data["stim_start"] = int(PNA['stim_start'] - signal_start)
    #     data["stim_end"] = int(PNA['stim_end'] - signal_start)
    #
    #     VNA_stim_response = savgol_filter(data["VNA"][data["stim_start"]:data["stim_end"]], 51, 1)
    #     HNA_stim_response = savgol_filter(data["HNA"][data["stim_start"]:data["stim_end"]], 51, 1)
    #     thr_HNA = np.quantile(HNA_stim_response, 0.65)
    #     thr_VNA = np.quantile(VNA_stim_response, 0.65)
    #     peak_HNA_inds = find_relevant_peaks(HNA_stim_response, threshold=thr_HNA, min_dist=150)
    #     peak_VNA_inds = find_relevant_peaks(VNA_stim_response, threshold=thr_VNA, min_dist=150)
    #
    #     fig = plot_chunk(data, y_lim = [1, 18])
    #     for i in range(len(fig.axes)):
    #         for j in range(len(peak_HNA_inds)):
    #             fig.axes[i].axvline(peak_HNA_inds[j] + stim_duration, color = 'purple')
    #         for j in range(len(peak_VNA_inds)):
    #             fig.axes[i].axvline(peak_VNA_inds[j] + stim_duration, color = 'green')
    #     fig.savefig(f"../img/experiments/traces/long_stim/{folder}")
    #     plt.close(fig)


    # combining_data("19032020", inds=[0, 1, 2]) # '2019-08-22_16-18-36_prc' - outlier

    # # PLOT PSD
    # data_folder = '../../data/sln_prc'
    # folder_save_to = '../../data/sln_prc_filtered'
    # folders = get_folders(data_folder, "prc")
    # folder = folders[1]
    # suffix = 'CH10'
    # signals = load(f'{data_folder}/{folder}/100_{suffix}.continuous', dtype=float)["data"]
    # plot_power_spectrum(signals, 30000)

    # # PLOT WITH MEANING OF THE PARAMETERS
    # num_rec = 3
    # num_chunk = 6
    # save_to = f'../../img/param_representation.png'
    # data = pickle.load(open(f'../../data/sln_prc_chunked/2019-09-05_12-26-14_prc/100_CH10_chunked.pkl', 'rb+'))
    # chunk = data[num_chunk]
    # clarifying_plot(chunk, save_to)



