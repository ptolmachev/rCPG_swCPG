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
ion()
from scipy.signal import butter, lfilter, freqz
from sklearn.covariance import EllipticEnvelope as EE
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.ensemble import IsolationForest as IF
from utils import *

def superplot(chunk):
    signal = chunk['PNA']
    i_starts = chunk['i_starts']
    i_ends = chunk['i_ends']
    stim = chunk['stim']
    fig = plt.figure(figsize = (15,5))
    plt.plot(signal, 'r', linewidth = 1, alpha = 0.95) # 46
    plt.axvline(stim + 47)
    plt.axvline(stim)
    for i in (i_starts):
        plt.axvline(i, color = 'g')
    for i in (i_ends):
        plt.axvline(i, color = 'k')
    plt.title(f'da')
    plt.grid(True)
    plt.show()

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def find_spikes(data):
    inds_above_threshold = []
    for n in range(data.shape[-1]):
        if data[n] > -4.95:
            inds_above_threshold.append(n)
    spikes_start = []
    spikes_start.append(inds_above_threshold[0])
    spikes_end = []
    i = 1
    while i < len(inds_above_threshold):
        if (inds_above_threshold[i] - inds_above_threshold[i - 1]) > 3:
            spikes_end.append(inds_above_threshold[i - 1])
            spikes_start.append(inds_above_threshold[i])
        i = i + 1
    spikes_end.append(inds_above_threshold[i - 1])
    return np.array(spikes_start), np.array(spikes_end)

def remove_spikes(rec, spikes_start, spikes_end):
    filtered_data = deepcopy(rec)
    for i in range(len(spikes_end)):
        t1 = spikes_start[i]-1
        t2 = spikes_end[i]+1
        filtered_data[t1:np.minimum(t2,rec.shape[-1])] = (filtered_data[t1] + filtered_data[t2-1])/2
    return filtered_data

def run_filtering():
    folders_all = os.listdir('../../data/sln_prc_preprocessed/')
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search("prc", str(folder))
        if m is not None:
            folders.append(folder)

    # first level processing
    suffixes = ['CH5', 'CH10', 'CH15']
    q = 0.95
    n = 0.8
    u = 1001
    conv_window = np.array([q ** i for i in np.arange(1, u)[::-1]]) / (np.sum(np.array([q ** i for i in np.arange(1, u)])))
    for folder in folders:
        print(folder)
        stim = pickle.load(open(f'../../data/sln_prc_preprocessed/{folder}/100_ADC1.pkl', 'rb+'))
        l = stim.shape[-1]
        stim = deepcopy(stim[:int(n * l)])
        spikes_start, spikes_end = find_spikes(stim)
        for suffix in suffixes:
            rec = pickle.load(open(f'../../data/sln_prc_preprocessed/{folder}/100_{suffix}.pkl', 'rb+'))[:int(n * l)]
            processed_signal = remove_spikes(rec, spikes_start, spikes_end)
            processed_signal = butter_highpass_filter(processed_signal, 300, 33000)
            processed_signal = np.convolve(np.abs(processed_signal), conv_window, 'full')
            processed_signal = savgol_filter(processed_signal, 1001, 3)
            processed_signal = processed_signal - np.mean(processed_signal)
            processed_signal = np.maximum(0.0, processed_signal)
            data = deepcopy(dict())
            data['stims_starts'] = spikes_start
            data['signal'] = processed_signal
            pickle.dump(data, open(f'../../data/sln_prc_preprocessed/{folder}/100_{suffix}_prc_processed.pkl', 'wb+'))
    return None

def combine_recordigs():
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
    pickle.dump(data_all, open(f'../../data/combined_data_prc_processed.pkl', 'wb+'))
    return None


def find_period(chunk):
    autocorr = np.correlate(chunk, chunk, mode='full')
    tmp = savgol_filter(autocorr[autocorr.shape[-1] // 2:], 151, 3)
    peaks_preliminary = find_peaks(tmp)[0]
    peaks = [0]
    for peak_candidate in peaks_preliminary:
        if (peak_candidate - peaks[-1]) > 100 and (tmp[peak_candidate] > 100):
            peaks.append(peak_candidate)

    peaks = np.array(peaks)
    Ts = peaks[1:] - peaks[:-1]
    T = np.mean(Ts)
    std_T = np.std(Ts)
    if len(peaks) > 1:
        return T, std_T
    else:
        return None, None

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

def chunk_data():
    data = pickle.load(open(f'../../data/combined_data_prc_processed.pkl', 'rb+'))
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
    pickle.dump(data_new, open(f'../../data/prc_data_chunked.pkl', 'wb+'))
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

def get_params(chunk):
    if not 'PNA' in list(chunk.keys()):
        print(chunk)
    signal = chunk['PNA']
    stim = chunk['stim']
    signal_filtered = savgol_filter(signal, 121, 1)
    threshold = 0.45
    signal_binary = binarise_signal(signal_filtered, threshold)
    signal_change = change(signal_binary)
    signal_begins = find_relevant_peaks(signal=signal_change, threshold=0.5).tolist()
    signal_ends = find_relevant_peaks(signal=-signal_change, threshold=0.5).tolist()

    #get rid of begins and ends if they are too close too stim:
    inds_to_del = []
    for i in range(len(signal_begins)):
        if np.abs(signal_begins[i] - stim) < 30 :
            inds_to_del.append(i)
    for ind in inds_to_del[::-1]:
        del signal_begins[ind]

    inds_to_del = []
    for i in range(len(signal_ends)):
        if np.abs(signal_ends[i] - stim) < 30 :
            inds_to_del.append(i)
    for ind in inds_to_del[::-1]:
        del signal_ends[ind]

    ts1, ts2, ts3, ts4, te1, te2, te3, te4 = get_onsets_and_ends(signal_begins, signal_ends, stim)

    Ti_0 = (te1 - ts1)
    T0 = (ts2 - ts1)
    Phi = (stim - ts2)
    Theta = (ts3 - stim)
    T1 = (ts3 - ts2)
    Ti_1 = (te3 - ts3)
    Ti_2 = (te4 - ts4)

    return Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2, ts1, ts2, ts3, ts4, te1, te2, te3, te4

def extract_data():
    # for all chunks in all recordings get an array of all these parameters:
    parameters_dict = {}
    data = pickle.load(open(f'../../data/prc_data_chunked.pkl', 'rb+'))
    for i, rec in enumerate(list(data.keys())):
        print(rec)
        if i in list(parameters_dict.keys()):
            pass
        else:
            parameters_dict[i] = []
        for num in list(data[rec].keys()):
            print(num)
            if data[rec][num] != {}:
                chunk = data[rec][num]
                # superplot(chunk)
                res = get_params(chunk)
                if not res is None:
                    parameters_dict[i].append(res)
    pickle.dump(parameters_dict, open(f'../../data/parameters_prc.pkl', 'wb+'))
    return None

def plot_interact(signal, stim_start, stim_end):
    pos = []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        pos.append(event.xdata)
        if len(pos) == 8:
            fig.canvas.mpl_disconnect(cid)
            close()
        plt.plot(x, y, 'ro')
        return None
    fig= plt.figure(figsize=(10,4))
    plot(signal, linewidth = 2, color='k')
    plt.axvline(stim_start)
    plt.axvline(stim_end)
    plt.grid(True)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    show(block=True)
    close()
    return pos

def extract_data_human_in_the_loop(dataset_chunks, save_to):
    parameters_dict = {}
    parameters_dict['count'] = 0
    parameters_dict['data'] = []
    print(f"The number of chunks of keys: {len(list(dataset_chunks.keys()))}")
    for num in list(dataset_chunks.keys()):
        print(f"chunk number: {num}")
        if dataset_chunks[num] != {}:
            chunk = dataset_chunks[num]
            # try to get parameters automatically:
            res_preliminary = get_params(chunk)
            Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2, ts1, ts2, ts3, ts4, te1, te2, te3, te4 = res_preliminary
            # prepare the chunk for data plot discarding unnecessary data
            if (not np.isnan(ts1)) and (not np.isnan(te4)):
                relevant_data_in_chunk = dict()
                relevant_data_in_chunk['VNA'] = chunk['VNA'][ts1:te4]
                relevant_data_in_chunk['PNA'] = chunk['PNA'][ts1:te4]
                relevant_data_in_chunk['HNA'] = chunk['HNA'][ts1:te4]
                relevant_data_in_chunk['T'] = chunk['T']
                relevant_data_in_chunk['std_T'] = chunk['std_T']
                relevant_data_in_chunk['stim'] = chunk['stim'] - ts1
                stim = relevant_data_in_chunk['stim']
                relevant_data_in_chunk['i_starts'] = (np.array(chunk['i_starts']) - ts1)[(np.array(chunk['i_starts']) >= ts1) * ((np.array(chunk['i_starts']) <= te4))]
                relevant_data_in_chunk['i_ends'] = (np.array(chunk['i_ends']) - ts1)[(np.array(chunk['i_ends']) >= ts1) * (np.array(chunk['i_ends']) <= te4)]

                ts = plot_interact(relevant_data_in_chunk['PNA'], relevant_data_in_chunk['stim'], relevant_data_in_chunk['stim'] + 50)
                if len(ts) == 8:
                    ts1, te1, ts2, te2, ts3, te3, ts4, te4 = ts
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
                pickle.dump(parameters_dict, open(save_to, 'wb+'))
                parameters_dict['count'] = parameters_dict['count'] + 1
                print(parameters_dict['count'])
    pickle.dump(parameters_dict, open(save_to, 'wb+'))
    return None

def fill_nans(data):
    df = pd.DataFrame(data)
    data_new = df.fillna(df.mean()).values
    return data_new


def get_rid_of_outliers(data):
    cov = IF(contamination=0.25).fit(data)
    mask = (cov.fit_predict(data) + 1) // 2
    inds = np.nonzero(mask)
    data_filtered = data[inds,:].squeeze()
    return data_filtered


if __name__ == '__main__':
    # run_filtering()
    # combine_recordigs()
    # chunk_data()
    # extract_data()
    # num_rec = 3
    # data = pickle.load(open(f'../../data/prc_data_chunked.pkl', 'rb+'))
    # dataset_chunks = data[list(data.keys())[num_rec]]
    # save_to = f'../../data/parameters_prc_08032020_{num_rec}.pkl'
    # extract_data_human_in_the_loop(dataset_chunks, save_to)
    # data = pickle.load(open('../../data/parameters_prc_allhuman.pkl','rb'))

    for i in range(4):
        num_rec = i
        data_ = pickle.load(open(f'../../data/parameters_prc_08032020_{num_rec}.pkl','rb'))['data']
        data_ = np.array(data_)
        data_ = fill_nans(data_)
        # data = get_rid_of_outliers(data)
        Phi = data_[:, 0]
        Ti_0 = data_[:, 1]
        T0 = data_[:, 2]
        T1 = data_[:, 3]
        Theta = data_[:, 4]
        Ti_1 = data_[:, 5]
        # Ti_2 = data[:, 6]
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
        plt.savefig(f"../../img/experiments/dataset_{num_rec + 1}_period_change.png")

        fig2 = plt.figure()
        plt.title("phase-cophase")
        y = cophase
        plt.scatter(phase, y)
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
        # plt.show()
        plt.savefig(f"../../img/experiments/dataset_{num_rec + 1}_phase_cophase.png")

        fig3 = plt.figure()
        y = T0
        plt.scatter(phase, y)
        plt.title("T0")
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
        # plt.show()
        plt.savefig(open(f"../../img/experiments/dataset_{num_rec + 1}_T0.png", "wb+"))

        fig4 = plt.figure()
        plt.title("T 1")
        y = T1
        plt.scatter(phase, y)
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
        # plt.show()
        plt.savefig(f"../../img/experiments/dataset_{num_rec + 1}_T1.png")

        fig5 = plt.figure()
        plt.title("Ti 0")
        y = Ti_0
        plt.scatter(phase, y)
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
        # plt.show()
        plt.savefig(f"../../img/experiments/dataset_{num_rec + 1}_Ti0.png")

        fig6 = plt.figure()
        plt.title("Ti 1")
        y = Ti_1
        plt.scatter(phase, y)
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
        # plt.show()
        plt.savefig(f"../../img/experiments/dataset_{num_rec + 1}_Ti1.png")

