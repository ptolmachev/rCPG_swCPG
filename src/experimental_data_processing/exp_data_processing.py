import sys
sys.path.insert(0, "../")
import signal
import pandas as pd
import numpy as np
import os
import re
import pickle
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from copy import deepcopy
from scipy import signal
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, close
from scipy.signal import butter, lfilter, freqz, decimate, convolve
from utils import *
from reading_experimental_data import *
ion()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a
#
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def filter_signal(signal,cutoff_fr_low, cutoff_fr_high, fr):
    processed_signal = signal - np.mean(signal)
    processed_signal = butter_bandpass_filter(processed_signal, cutoff_fr_low, cutoff_fr_high, fr)
    return processed_signal

def smooth_signal(signal, window_len, sigma):
    signal = np.abs(signal)
    x = np.exp(((-(np.arange(window_len) - window_len // 2) ^ 2) / (2 * sigma ** 2))) #gaussian kernel
    kernel = x / (np.sum(x))
    signal = convolve(signal, kernel, 'same')
    return signal

def find_stim_spikes(stim, threshold=-4.95):
    '''finds sudden jumps in the applied stimulus data'''
    inds_above_threshold = np.where(stim > threshold)[0]
    tmp_e = np.diff(inds_above_threshold)
    ends = np.array(inds_above_threshold[np.where(tmp_e > 1)]) + 1  #indices where the stimulus starts

    inds_below_threshold = np.where(stim < threshold)[0]
    tmp_s = np.diff(inds_below_threshold)
    starts = np.array(inds_below_threshold[np.where(tmp_s > 1)]) #indices where the stimulus ends
    return starts, ends

def remove_stim_spikes(rec, spikes_start, spikes_end, offset):
    # interpolates data
    filtered_data = deepcopy(rec)
    for i in range(len(spikes_end)):
        t1 = spikes_start[i] - offset
        t2 = spikes_end[i] + offset
        filtered_data[t1:t2] = (filtered_data[t1] + filtered_data[t2])/2.0
    return filtered_data

def get_folders(root_folder, pattern):
    folders_all = os.listdir(root_folder + '/')
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search(pattern, str(folder))
        if m is not None:
            folders.append(folder)
    return folders

def run_filtering(data_folder, folder_save_to):
    folders = get_folders(data_folder, "prc")
    suffixes = ['CH5', 'CH10', 'CH15'] #
    fr = 30000
    cutoff_fr_high = 3000
    cutoff_fr_low = 300
    stim_spike_threshold = -4.95
    offset=100
    smoothing_window=1000
    sigma=100
    downsampling_factor=10
    for folder in folders:
        print(folder)
        stim = load(f'{data_folder}/{folder}/100_ADC1.continuous', dtype=float)["data"]
        stim_spikes_start, stim_spikes_end = find_stim_spikes(stim, stim_spike_threshold)

        # cluster stim-related spikes into one stimulus
        stim_start = stim_spikes_start[np.array(np.where(np.diff(stim_spikes_start) > 2500)) - 5].squeeze() / downsampling_factor ** 2
        stim_end = stim_spikes_end[np.where(np.diff(stim_spikes_end) > 2500)] / downsampling_factor ** 2

        for suffix in suffixes:
            rec = load(f'{data_folder}/{folder}/100_{suffix}.continuous', dtype=float)["data"]
            #filter signal
            processed_signal = filter_signal(rec, cutoff_fr_low, cutoff_fr_high, fr)
            #despike signal
            despiked_procesed_signal = remove_stim_spikes(processed_signal, stim_spikes_start, stim_spikes_end, offset=offset)
            #smooth signal
            smoothed_signal = smooth_signal(despiked_procesed_signal, window_len=smoothing_window, sigma=sigma)
            # downsample signal
            signal = decimate(decimate(smoothed_signal, downsampling_factor), downsampling_factor)
            #save_data
            data_to_save = {}
            data_to_save['stim_start'] = stim_start
            data_to_save['stim_end'] = stim_end
            data_to_save['signal'] = signal
            pickle.dump(data_to_save, open(f'{folder_save_to}/{folder}/100_{suffix}_processed.pkl', 'wb+'))
    return None

def split_signal_into_chunks(signal, stim_starts):
    # for span between the two stims
    data_chunked = dict()
    for i in range(len(stim_starts) - 1):
        print(f'chunk number {i}')
        start = stim_starts[i]
        end = stim_starts[i + 1]
        chunk = signal[int(start):int(end)]
        # find period
        T, std_T = get_period(chunk)
        new_chunk = signal[int(start + int(1 * T)):int(end + int(4 * T))]
        data_chunked[i] = dict()
        data_chunked[i]['signal'] = new_chunk
        data_chunked[i]['T'] = T
        data_chunked[i]['std_T'] = std_T
        data_chunked[i]['stim'] = new_chunk.shape[-1] - int(4 * T)
        # i_start, i_end = get_insp_phases(data_chunked[i]['signal'])
        # data_chunked[i]['i_starts'] = i_start
        # data_chunked[i]['i_ends'] = i_end
    return data_chunked

def chunk_data(data_folder, save_to):
    '''splits huge recording into chunks with one stimulus per chunk'''
    folders = get_folders(data_folder, "_prc")
    suffix = 'CH10'
    for folder in folders:
        print(folder)
        data = pickle.load(open(f'{data_folder}/{folder}/100_{suffix}_processed.pkl', 'rb+'))
        signal = data['signal']
        stim_start = data['stim_start']
        stim_end = data['stim_end']
        data_chunked = split_signal_into_chunks(signal, stim_start)
        pickle.dump(data_chunked, open(f'{save_to}/{folder}/100_{suffix}_chunked.pkl', 'wb+'))
    return None

def get_onsets_and_ends(signal_begins, signal_ends, stim):
    try:
        ts2 = last_lesser_than(signal_begins, stim)[0]
    except:
        ts2 = np.nan
    try:
        ts3 = first_greater_than(signal_begins, stim)[0]
    except:
        ts3 = np.nan
    try:
        ts1 = last_lesser_than(signal_begins, ts2)[0]
    except:
        ts1 = np.nan
    try:
        ts4 = first_greater_than(signal_begins, ts3)[0]
    except:
        ts4 = np.nan
    try:
        te1 = first_greater_than(signal_ends, ts1)[0]
    except:
        te1 = np.nan
    try:
        te2 = first_greater_than(signal_ends, ts2)[0]
    except:
        te2 = np.nan
    try:
        te3 = first_greater_than(signal_ends, ts3)[0]
    except:
        te3 = np.nan
    try:
        te4 = first_greater_than(signal_ends, ts4)[0]
    except:
        te4 = np.nan
    return ts1, ts2, ts3, ts4, te1, te2, te3, te4


def get_inspiration_onsets_and_ends(signal, threshold):
    signal_binary = binarise_signal(signal, threshold)
    signal_change = change(signal_binary)
    signal_begins = find_relevant_peaks(signal=signal_change, threshold=0.5).tolist()
    signal_ends = find_relevant_peaks(signal=-signal_change, threshold=0.5).tolist()
    return signal_begins, signal_ends

def get_timings(insp_begins, insp_ends, stim):
    timings = {}
    timings['t_start'] = {}
    timings['t_end'] = {}
    ind_insp_0 = np.searchsorted(insp_begins, stim) - 1
    for i in range(ind_insp_0+1):
        timings['t_start'][-i] = insp_begins[ind_insp_0-i]
        timings['t_end'][-i] = insp_ends[np.searchsorted(insp_ends, timings['t_start'][-i])]

    for i in range(len(insp_begins) - ind_insp_0):
        timings['t_start'][i] = insp_begins[ind_insp_0+i]
        timings['t_end'][i] = insp_ends[np.searchsorted(insp_ends, timings['t_start'][i])]
    return timings

def extract_data_auto(dataset_chunks, save_to):
    parameters_dict = {}
    parameters_dict['count'] = 0
    parameters_dict['data'] = []
    print(f"The number of chunks of keys: {len(list(dataset_chunks.keys()))}")
    for num in list(dataset_chunks.keys()):
        print(f"chunk number: {num}")
        chunk = dataset_chunks[num]
        PNA = chunk['signal']
        stim = chunk['stim']
        insp_begins, insp_ends = get_inspiration_onsets_and_ends(PNA, threshold=7.5)
        ts = get_timings(insp_begins, insp_ends, stim)
        ind_neg_starts = np.where(np.array(list((ts["t_start"].keys()))) < 0)[0]
        neg_starts = []
        for i in range(len(ind_neg_starts)):
            neg_starts.append(ts['t_start'][list(ts['t_start'].keys())[ind_neg_starts[i]]])
        neg_starts = np.array(neg_starts)[::-1]

        ind_neg_end = np.where(np.array(list((ts["t_end"].keys()))) < 0)[0]
        neg_ends = []
        for i in range(len(ind_neg_end)):
            neg_ends.append(ts['t_end'][list(ts['t_end'].keys())[ind_neg_end[i]]])
        neg_ends = np.array(neg_ends)[::-1]

        Phi = stim - ts["t_start"][0]
        Ti_0 = np.mean(neg_ends-neg_starts)
        T0 = np.mean(np.diff(neg_starts))
        T1 = ts["t_start"][1] - ts["t_start"][0]
        Theta = ts["t_start"][1] - stim
        Ti_1 = ts["t_end"][1] - ts["t_start"][1]
        Ti_2 = ts["t_end"][2] - ts["t_start"][2]
        res = (Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2)
        print(res)
        parameters_dict['data'].append(res)
        # dump after every point
        parameters_dict['count'] = parameters_dict['count'] + 1
        pickle.dump(parameters_dict, open(save_to, 'wb+'))
        print(parameters_dict['count'])
    return None

def plot_interact(signal, stim_start, stim_end):
    '''plots the signal and allows to specify begins and ends of inspiratory phases manually'''
    pos = []

    def draw(signal, stim_start, stim_end):
        fig = plt.figure(figsize=(20, 6))
        plot(signal, linewidth=2, color='k')
        plt.axvline(stim_start)
        plt.axvline(stim_end)
        plt.grid(True)
        return fig

    def set_handlers(fig):
        cid = fig.canvas.mpl_connect('button_press_event', get_x)
        cid_exit = fig.canvas.mpl_connect('key_press_event', exit)
        return cid, cid_exit

    def get_x(event):
        x = event.xdata
        y = event.ydata
        pos.append(event.xdata)
        plt.plot(x, y, 'ro')
        return None

    def exit(event):
        if event.key == 'e':
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.mpl_disconnect(cid_exit)
            close()
        return None

    fig = draw(signal, stim_start, stim_end)
    cid, cid_exit = set_handlers(fig)
    show(block=True)
    close()
    # we need only the last 8 positions to caclulate parameters: ts1, ts2, ts3, ts4, te1, te2, te3, te4
    return pos[-8:]

def extract_data_human_in_the_loop(dataset_chunks, save_to):
    parameters_dict = {}
    parameters_dict['count'] = 0
    parameters_dict['data'] = []
    print(f"The number of chunks of keys: {len(list(dataset_chunks.keys()))}")
    for num in list(dataset_chunks.keys()):
        print(f"chunk number: {num}")
        chunk = dataset_chunks[num]
        PNA = chunk['signal']
        stim = chunk['stim']
        threshold = 0.1
        max_length = 50
        ts = plot_interact(PNA, stim, stim + 50)
        if len(ts) == 8:
            ts1, te1, ts2, te2, ts3, te3, ts4, te4 = ts
            if (not np.isnan(ts1)) and (not np.isnan(te4)):
                Ti_0 = (te1 - ts1)
                T0 = (ts2 - ts1)
                Phi = (stim - ts2)
                Theta = (ts3 - stim)
                T1 = (ts3 - ts2)
                Ti_1 = (te3 - ts3)
                Ti_2 = (te4 - ts4)
                res = (Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2)
                print(res)
                parameters_dict['data'].append(res)
                # dump after every point
                parameters_dict['count'] = parameters_dict['count'] + 1
                pickle.dump(parameters_dict, open(save_to, 'wb+'))
                print(parameters_dict['count'])
    return None

def plot_final_data(num_rec, file_load, dir_save_to):
    data_ = pickle.load(open(file_load,'rb'))['data']
    data_ = np.array(data_)
    data_ = fill_nans(data_)
    # data = get_rid_of_outliers(data)
    Phi = data_[:, 0]
    Ti_0 = data_[:, 1]
    T0 = data_[:, 2]
    T1 = data_[:, 3]
    Theta = data_[:, 4]
    Ti_1 = data_[:, 5]
    Ti_2 = data_[:, 6]
    phase = np.minimum(1, 1 * (Phi/T0))
    cophase = np.minimum(1, 1 * (Theta/T0))

    fig1 = plt.figure()
    plt.title("T1/T0")
    y = T1/T0
    plt.scatter(phase, y)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    # plt.show()
    plt.savefig(f"{dir_save_to}dataset_{num_rec + 1}_period_change.png")

    fig2 = plt.figure()
    plt.title("phase-cophase")
    y = cophase
    plt.scatter(phase, y)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.show()
    plt.savefig(f"{dir_save_to}dataset_{num_rec + 1}_phase_cophase.png")

    fig3 = plt.figure()
    y = T0
    plt.scatter(phase, y)
    plt.title("T0")
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    # plt.show()
    plt.savefig(open(f"{dir_save_to}dataset_{num_rec + 1}_T0.png", "wb+"))

    fig4 = plt.figure()
    plt.title("T 1")
    y = T1
    plt.scatter(phase, y)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    # plt.show()
    plt.savefig(f"{dir_save_to}dataset_{num_rec + 1}_T1.png")

    fig5 = plt.figure()
    plt.title("Ti 0")
    y = Ti_0
    plt.scatter(phase, y)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    # plt.show()
    plt.savefig(f"{dir_save_to}dataset_{num_rec + 1}_Ti0.png")

    fig6 = plt.figure()
    plt.title("Ti 1")
    y = Ti_1
    plt.scatter(phase, y)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    # plt.show()
    plt.savefig(f"{dir_save_to}dataset_{num_rec + 1}_Ti1.png")

    fig7 = plt.figure()
    plt.title("Ti 2")
    y = Ti_2
    plt.scatter(phase, y)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    # plt.show()
    plt.savefig(f"{dir_save_to}dataset_{num_rec + 1}_Ti2.png")
    return None

def clarifying_plot(chunk, save_to):
    PNA = chunk['PNA']
    s = int(0.58 * len(PNA))
    e = int(0.93 * len(PNA))
    PNA = PNA[s:e]
    stim = chunk['stim'] - s
    fig = plt.figure(figsize=(20, 6))
    plt.plot(PNA, linewidth=3, color='k')

    ts = get_times_auto(binarise_signal(PNA, 0.8), stim)
    ts1, ts2, ts3, ts4, te1, te2, te3, te4 = ts

    plt.axvline(stim, color='r', linestyle='-')
    plt.axvline(stim + 50, color='r', linestyle='-')
    plt.axvspan(stim, stim + 50, alpha=0.25, color='red')

    plt.axvline(ts1, color='b', linestyle='--')
    plt.axvline(ts2, color='b', linestyle='--')
    plt.axvline(ts3, color='b', linestyle='--')
    plt.axvline(ts4, color='b', linestyle='--')
    plt.axvline(te1, color='b', linestyle='--')
    plt.axvline(te2, color='b', linestyle='--')
    plt.axvline(te3, color='b', linestyle='--')
    plt.axvline(te4, color='b', linestyle='--')

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

    margin = 5
    ConvergingArrows(stim, stim+50, 1.65, margin)  # Stim
    DoubleArrow(ts1, ts2, 1.55, margin) # T0
    DoubleArrow(ts2, ts3, 1.55, margin)  # T1
    DoubleArrow(ts1, te1, 0.3, margin)  # Ti_0
    DoubleArrow(ts3, te3, 0.3, margin)  # Ti_1
    DoubleArrow(ts4, te4, 0.3, margin)  # Ti_2
    DoubleArrow(ts2, stim, 0.2, margin)  # Phi
    DoubleArrow(stim+50, ts3, 0.2, margin)  # Theta

    plt.title("Phrenic Nerve Activity", fontsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0.15, 1.75])
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    # fig.patch.set_visible(False)
    # plt.show(block=True)
    plt.axis('off')
    plt.savefig(f"{save_to}")
    plt.close()
    return None

if __name__ == '__main__':
    # data_folder = '../../data/sln_prc'
    # folder_save_to = '../../data/sln_prc_filtered'
    # run_filtering(data_folder, folder_save_to)

    # # split data into chunks
    # data_folder = f'../../data/sln_prc_filtered'
    # save_to = f'../../data/sln_prc_chunked'
    # chunk_data(data_folder, save_to)

    file = '100_CH10_chunked.pkl'
    folder = '2019-08-22_16-18-36_prc'
    data_folder = f'../../data/sln_prc_chunked'
    data = pickle.load(open(f'{data_folder}/{folder}/{file}', 'rb+'))
    save_to = f'../../data/parameters_prc_17032020_{0}.pkl'
    extract_data_auto(data, save_to)

    # # plotting final data
    # num_rec = 0
    # file_load = f'../../data/parameters_prc_17032020_{num_rec}.pkl'
    # dir_save_to = '../../img/experiments/'
    # plot_final_data(num_rec, file_load, dir_save_to)

    # plotting parameter representation
    # num_rec = 3
    # num_chunk = 6
    # save_to = f'../../img/param_representation.png'
    # data = pickle.load(open(f'../../data/prc_data_chunked.pkl', 'rb+'))
    # dataset_chunks = data[list(data.keys())[num_rec]]
    # chunk = dataset_chunks[num_chunk]
    # clarifying_plot(chunk, save_to)
