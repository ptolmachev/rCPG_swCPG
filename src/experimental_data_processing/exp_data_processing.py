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
from scipy.signal import butter, lfilter, freqz
from utils import *
ion()

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def remove_spikes(signal, threshold, max_length):
    binarised_signal = binarise_signal(signal, threshold)
    change = np.diff(binarised_signal)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    #make sure start goes before end
    if starts[0] > ends[0]:
        starts = np.append(0, starts)
    #make sure the end is the last
    if ends[-1] < starts[-1]:
        ends = np.append(ends, len(binarised_signal))

    new_signal = deepcopy(signal)
    lengths = ends - starts
    for i in range(len(lengths)):
        if lengths[i] <= max_length:
            s = np.maximum(starts[i]-1, 0)
            e = np.minimum(ends[i]+1, len(signal)-1)
            new_signal[s:e] = (signal[s] + signal[e]) / 2
    return new_signal

def remove_dips(signal, threshold, max_length):
    binarised_signal = binarise_signal(signal, threshold)
    change = np.diff(binarised_signal)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    #make sure ends goes before stars
    if ends[0] > starts[0]:
        ends = np.append(0, ends)
    #make sure the end is the last
    if starts[-1] < ends[-1]:
        starts = np.append(starts, len(binarised_signal))

    new_signal = deepcopy(signal)
    lengths = starts - ends
    for i in range(len(lengths)):
        if lengths[i] <= max_length:
            new_signal[starts[i]-1:ends[i]+1] = (signal[starts[i] - 1] + signal[ends[i] + 1]) / 2
    return new_signal


def find_spikes(data, threshold, length):
    '''finds sudden jumps in the data'''
    inds_above_threshold = []
    for n in range(data.shape[-1]):
        if data[n] > threshold:
            inds_above_threshold.append(n)
    spikes_start = []
    spikes_start.append(inds_above_threshold[0])
    spikes_end = []
    i = 1
    while i < len(inds_above_threshold):
        if (inds_above_threshold[i] - inds_above_threshold[i - 1]) > length:
            spikes_end.append(inds_above_threshold[i - 1])
            spikes_start.append(inds_above_threshold[i])
        i = i + 1
    spikes_end.append(inds_above_threshold[i - 1])
    return np.array(spikes_start), np.array(spikes_end)
#
# def remove_spikes(rec, spikes_start, spikes_end):
#     filtered_data = deepcopy(rec)
#     for i in range(len(spikes_end)):
#         t1 = spikes_start[i]-1
#         t2 = spikes_end[i]+1
#         filtered_data[t1:np.minimum(t2,rec.shape[-1])] = (filtered_data[t1] + filtered_data[t2-1])/2
#     return filtered_data

# def filtering_test(data_folder):
#     folders_all = os.listdir(data_folder + '/')
#     folders = []
#     for i, folder in enumerate(folders_all):
#         m = re.search("prc", str(folder))
#         if m is not None:
#             folders.append(folder)
#     # first level processing
#     suffixes = ['CH5', 'CH10', 'CH15']
#     n = 0.15
#     u = 10001
#     fr = 300
#     cutoff_fr_high = 3
#     for folder in folders:
#         print(folder)
#         stim = pickle.load(open(f'{data_folder}/{folder}/100_ADC1.pkl', 'rb+'))
#         l = stim.shape[-1]
#         stim = deepcopy(stim[:int(n * l)])
#         # spikes_start, spikes_end = find_spikes(stim, -4.95, 3)
#         for suffix in suffixes:
#             if suffix == "CH10":
#                 rec = pickle.load(open(f'{data_folder}/{folder}/100_{suffix}.pkl', 'rb+'))[:int(n * l)]
#                 processed_signal = remove_spikes(rec, -4.95, 5)
#                 processed_signal = processed_signal - np.mean(processed_signal)
#                 processed_signal = butter_highpass_filter(processed_signal, cutoff_fr_high, fr)
#                 processed_signal = np.abs(processed_signal)
#                 processed_signal = savgol_filter(processed_signal, u, 3)
#     return None

def run_filtering(data_folder):
    # first level processing
    folders_all = os.listdir(data_folder + '/')
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search("prc", str(folder))
        if m is not None:
            folders.append(folder)
    suffixes = ['CH5', 'CH10', 'CH15']
    n = 0.85
    u = 10001
    fr = 300
    cutoff_fr_high = 3
    for folder in folders:
        print(folder)
        stim = pickle.load(open(f'{data_folder}/{folder}/100_ADC1.pkl', 'rb+'))
        l = stim.shape[-1]
        stim = deepcopy(stim[:int(n * l)])
        spikes_start, spikes_end = find_spikes(stim, -4.95, 5)
        for suffix in suffixes:
            rec = pickle.load(open(f'{data_folder}/{folder}/100_{suffix}.pkl', 'rb+'))[:int(n * l)]
            processed_signal = remove_spikes(rec, -4.95, 5)
            processed_signal = processed_signal - np.mean(processed_signal)
            processed_signal = butter_highpass_filter(processed_signal, cutoff_fr_high, fr)
            processed_signal = np.abs(processed_signal)
            processed_signal = savgol_filter(processed_signal, u, 3)
            data = deepcopy(dict())
            data['stims_starts'] = spikes_start
            data['signal'] = processed_signal
            pickle.dump(data, open(f'{data_folder}/{folder}/100_{suffix}_prc_processed.pkl', 'wb+'))
    return None

def combine_recordigs(save_to):
    '''collects all the recordings in one dataset'''
    folders_all = os.listdir('../../data/sln_prc_preprocessed/')
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search("_prc", str(folder))
        if m is not None:
            folders.append(folder)
    # second level processing
    suffixes = ['CH5', 'CH10', 'CH15']
    wd = 3500
    down_factor = 100
    data_all = dict()
    for folder in folders:
        print(folder)
        data_to_save = dict()
        for suffix in suffixes:
            data = pickle.load(open(f'../../data/sln_prc_preprocessed/{folder}/100_{suffix}_prc_processed.pkl', 'rb+'))
            # downsample
            data['signal'] = pd.DataFrame(data['signal']).rolling(window=wd).mean().dropna().values.squeeze()
            data['signal'] = data['signal'][::down_factor]
            data['stims_starts'] = np.round(data['stims_starts'] / down_factor, 0)
            if suffix == 'CH5':
                data_to_save['HNA'] = data['signal']
            elif suffix == 'CH10':
                data_to_save['PNA'] = data['signal']
            elif suffix == 'CH15':
                data_to_save['VNA'] = data['signal']

            stims_clustered = [0]
            # cluster them and save again:
            for i in range(len(data['stims_starts'])):
                if ((np.abs(data['stims_starts'][i] - stims_clustered[-1])) > 100):
                    stims_clustered.append(data['stims_starts'][i])

            data_to_save['stims_starts'] = stims_clustered
            data_all[folder] = deepcopy(data_to_save)
    pickle.dump(data_all, open(save_to, 'wb+'))
    return None

def get_insp_phases(chunk):
    signal = chunk['PNA']
    stim = chunk['stim']
    signal_filtered = savgol_filter(signal, 121, 1)
    threshold = 0.35
    signal_binary = binarise_signal(signal_filtered, threshold)
    signal_change = change(signal_binary)
    begins_inds = find_relevant_peaks(signal_change, 0.5)
    ends_inds = find_relevant_peaks(-signal_change, 0.5)
    return begins_inds, ends_inds

def chunk_data(load_file, save_to):
    '''splits huge recording into chunks with one stimulus per chunk'''
    data = pickle.load(open(load_file, 'rb+'))
    # for recording in data
    data_new = dict()
    for rec in list(data.keys()):
        print(rec)
        data_new[rec] = dict()
        stims = data[rec]['stims_starts']
        # for span between the two stims
        for i in range(len(stims) - 1):
            print(f'chunk number {i}')
            data_new[rec][i] = dict()
            start = stims[i]
            end = stims[i + 1]
            chunk = data[rec]['VNA'][int(start):int(end)]
            # find period
            T, std_T = get_period(chunk)
            new_chunk_VNA = data[rec]['VNA'][int(start + int(1 * T)):int(end + int(3 * T))]
            new_chunk_PNA = data[rec]['PNA'][int(start + int(1 * T)):int(end + int(3 * T))]
            new_chunk_HNA = data[rec]['HNA'][int(start + int(1 * T)):int(end + int(3 * T))]

            data_new[rec][i] = dict()
            data_new[rec][i]['VNA'] = new_chunk_VNA
            data_new[rec][i]['PNA'] = new_chunk_PNA
            data_new[rec][i]['HNA'] = new_chunk_HNA
            data_new[rec][i]['T'] = T
            data_new[rec][i]['std_T'] = std_T
            data_new[rec][i]['stim'] = new_chunk_VNA.shape[-1] - int(3 * T)
            i_start, i_end = get_insp_phases(data_new[rec][i])
            data_new[rec][i]['i_starts'] = i_start
            data_new[rec][i]['i_ends'] = i_end

            # fig = plt.figure(figsize=(9,5))
            # plt.plot(data_new[rec][i]['VNA'], 'k')
            # plt.axvline(data_new[rec][i]['stim'] )
            # show(block=True)
            # close()

    pickle.dump(data_new, open(save_to, 'wb+'))
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

def get_times_auto(signal, stim):
    ''' automatically extracts ts1, ts2, ts3, ts4, te1, te2, te3, te4.'''
    threshold = 0.4
    signal_binary = binarise_signal(signal, threshold)
    signal_change = change(signal_binary)
    signal_begins = find_relevant_peaks(signal=signal_change, threshold=0.5).tolist()
    signal_ends = find_relevant_peaks(signal=-signal_change, threshold=0.5).tolist()

    ts1, ts2, ts3, ts4, te1, te2, te3, te4 = get_onsets_and_ends(signal_begins, signal_ends, stim)
    return ts1, ts2, ts3, ts4, te1, te2, te3, te4

def plot_interact(signal, stim_start, stim_end):
    '''plots the signal and allows to specify begins and ends of inspiratory phase'''
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
        if dataset_chunks[num] != {}:
            chunk = dataset_chunks[num]
            PNA = chunk['PNA']
            stim = chunk['stim']
            # apply thresholding!
            # a = np.zeros_like(PNA)
            # a[np.where(PNA < np.quantile(PNA, 0.8))] = True
            # PNA[a.astype(bool)] = 0
            # remove spikes
            threshold = 0.1
            max_length = 50
            # PNA = remove_spikes(PNA, threshold, max_length)
            # PNA = remove_dips(PNA, threshold, 100)
            # try to get parameters automatically:
            # ts = get_times_auto(PNA, stim)
            # ts1, ts2, ts3, ts4, te1, te2, te3, te4 = ts
            # # prepare the chunk for data plot discarding unnecessary data
            # if (not np.isnan(ts1)) and (not np.isnan(te4)):
            #     relevant_data_in_chunk = dict()
            #     relevant_PNA = PNA[ts1:te4]
            #     # relevant_data_in_chunk['VNA'] = chunk['VNA'][ts1:te4]
            #     # relevant_data_in_chunk['PNA'] = chunk['PNA'][ts1:te4]
            #     # relevant_data_in_chunk['HNA'] = chunk['HNA'][ts1:te4]
            #     relevant_data_in_chunk['T'] = chunk['T']
            #     relevant_data_in_chunk['std_T'] = chunk['std_T']
            #     relevant_data_in_chunk['stim'] = chunk['stim'] - ts1
            #     stim = relevant_data_in_chunk['stim']
            #     relevant_data_in_chunk['i_starts'] = (np.array(chunk['i_starts']) - ts1)[(np.array(chunk['i_starts']) >= ts1) * ((np.array(chunk['i_starts']) <= te4))]
            #     relevant_data_in_chunk['i_ends'] = (np.array(chunk['i_ends']) - ts1)[(np.array(chunk['i_ends']) >= ts1) * (np.array(chunk['i_ends']) <= te4)]
            #
            #     ts = plot_interact(relevant_PNA, stim, stim + 50)
            #     if len(ts) == 8:
            #         ts1, te1, ts2, te2, ts3, te3, ts4, te4 = ts
            #         Ti_0 = (te1 - ts1)
            #         T0 = (ts2 - ts1)
            #         Phi = (stim - ts2)
            #         Theta = (ts3 - stim)
            #         T1 = (ts3 - ts2)
            #         Ti_1 = (te3 - ts3)
            #         Ti_2 = (te4 - ts4)
            #         res = (Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2)
            #         print(res)
            #         parameters_dict['data'].append(res)
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


    margin = 100
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
    DoubleArrow(ts2, te2, 0.3, margin)  # Ti_1
    DoubleArrow(ts3, te3, 0.3, margin)  # Ti_2

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

    # data_folder = '../../data/sln_prc_preprocessed'
    # run_filtering(data_folder)

    # save_to = f'../../data/combined_data_prc_processed.pkl'
    # combine_recordigs(save_to)

    # split data into chunks
    # load_file = f'../../data/combined_data_prc_processed.pkl'
    # save_to = f'../../data/prc_data_chunked.pkl'
    # chunk_data(load_file, save_to)

    # num_rec = 2
    # data = pickle.load(open(f'../../data/prc_data_chunked.pkl', 'rb+'))
    # dataset_chunks = data[list(data.keys())[num_rec]]
    # save_to = f'../../data/parameters_prc_12032020_{num_rec}.pkl'
    # extract_data_human_in_the_loop(dataset_chunks, save_to)

    # # plotting final data
    num_rec = 2
    file_load = f'../../data/parameters_prc_12032020_{num_rec}.pkl'
    dir_save_to = '../../img/experiments/'
    plot_final_data(num_rec, file_load, dir_save_to)

    # plotting parameter representation
    # num_rec = 3
    # num_chunk = 6
    # save_to = f'../../img/param_representation.png'
    # data = pickle.load(open(f'../../data/prc_data_chunked.pkl', 'rb+'))
    # dataset_chunks = data[list(data.keys())[num_rec]]
    # chunk = dataset_chunks[num_chunk]
    # clarifying_plot(chunk, save_to)
