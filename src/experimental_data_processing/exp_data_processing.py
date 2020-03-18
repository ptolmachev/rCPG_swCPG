import sys
sys.path.insert(0, "../")
import numpy as np
import pickle
from copy import deepcopy
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, close
from utils import *
from reading_experimental_data import *
from scipy.signal import decimate, convolve
ion()

def filter_signal(signal,cutoff_fr_low, cutoff_fr_high, fr):
    processed_signal = signal - np.mean(signal)
    processed_signal = butter_bandpass_filter(processed_signal, cutoff_fr_low, cutoff_fr_high, fr)
    return processed_signal

def smooth_signal(signal, window_len, sigma):
    signal = np.abs(signal)
    x = np.exp(((-(np.arange(window_len) - window_len // 2) ^ 2) / (2 * sigma ** 2))) # gaussian kernel
    kernel = x / (np.sum(x)) #normalise window to add up to one
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

def run_filtering(data_folder, folder_save_to):
    folders = get_folders(data_folder, "prc")
    suffixes = ['CH5', 'CH10', 'CH15'] #
    fr = 30000
    cutoff_fr_high = 3000
    cutoff_fr_low = 300
    stim_spike_threshold = -4.95
    offset=100
    smoothing_window=1000
    sigma=150
    downsampling_factor=10
    for folder in folders:
        print(folder)
        stim = load(f'{data_folder}/{folder}/100_ADC1.continuous', dtype=float)["data"]
        stim_spikes_start, stim_spikes_end = find_stim_spikes(stim, stim_spike_threshold)

        # cluster stim-related spikes into one stimulus
        stim_start = stim_spikes_start[np.array(np.where(np.diff(stim_spikes_start) > 2500)) - 5].squeeze() / downsampling_factor ** 2
        stim_end = stim_spikes_end[np.where(np.diff(stim_spikes_end) > 2500)].squeeze() / downsampling_factor ** 2

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

def split_signal_into_chunks(signal, stim_starts, stim_end):
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
        data_chunked[i]['stim_start'] = new_chunk.shape[-1] - int(4 * T)
        data_chunked[i]['stim_end'] = new_chunk.shape[-1] + int((stim_end[i] - stim_starts[i])) - int(4 * T)
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
        data_chunked = split_signal_into_chunks(signal, stim_start, stim_end)
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


def get_inspiration_onsets_and_ends(signal, threshold, min_len):
    signal_binary = binarise_signal(signal, threshold)
    signal_change = np.diff(signal_binary)
    signal_begins = find_relevant_peaks(signal=signal_change, threshold=0.5)
    signal_ends = find_relevant_peaks(signal=-signal_change, threshold=0.5)

    x = signal_ends[0] - signal_begins[0]
    if x < 0:
        signal_ends = signal_ends[1:]

    if len(signal_ends) != len(signal_begins):
        signal_begins = signal_begins[:np.minimum(len(signal_ends), len(signal_begins))]
        signal_ends = signal_ends[:np.minimum(len(signal_ends), len(signal_begins))]

    #filter signal_begins
    irrelevant_inds_begins = np.array(signal_begins)[np.where(signal_ends - signal_begins < min_len)[0]]
    irrelevant_inds_ends = np.array(signal_ends)[np.where(signal_ends - signal_begins < min_len)[0]]

    signal_begins = [elem for elem in signal_begins if not elem in irrelevant_inds_begins]
    signal_ends = [elem for elem in signal_ends if not elem in irrelevant_inds_ends]

    return signal_begins, signal_ends

def get_timings(insp_begins, insp_ends, stim, len_chunk):
    timings = {}
    timings['t_start'] = {}
    timings['t_end'] = {}
    ind_insp_0 = np.searchsorted(insp_begins, stim) - 1
    for i in range(ind_insp_0+1):
        timings['t_start'][-i] = insp_begins[ind_insp_0-i]
        timings['t_end'][-i] = insp_ends[np.searchsorted(insp_ends, timings['t_start'][-i])]

    for i in range( np.minimum(len(insp_begins), len(insp_ends)) - ind_insp_0):
        timings['t_start'][i] = insp_begins[ind_insp_0+i]
        ind = np.searchsorted(insp_ends, timings['t_start'][i])
        timings['t_end'][i] = insp_ends[ind] if ind != len(insp_ends) else len_chunk - 1
    return timings

def extract_data_from_chunk(dataset_chunks, save_to):
    threshold = 6.5 #value of signal at which the inspiration is detected
    min_len = 50 # if len between start of insp and end of insp is lesser than min_len - discard
    parameters_dict = {}
    parameters_dict['count'] = 0
    parameters_dict['data'] = []
    print(f"The number of chunks of keys: {len(list(dataset_chunks.keys()))}")
    for num in list(dataset_chunks.keys()):
        print(f"chunk number: {num}")
        chunk = dataset_chunks[num]
        PNA = chunk['signal']
        stim_start = chunk['stim_start']
        stim_end = chunk['stim_end']
        insp_begins, insp_ends = get_inspiration_onsets_and_ends(PNA, threshold, min_len)
        len_chunk = len(PNA)
        ts = get_timings(insp_begins, insp_ends, stim_start, len_chunk)

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

        # plot for checking

        # plt.figure(figsize=(15, 5))
        # plt.plot(PNA, linewidth=2, color="k")
        # plt.grid(True)
        # plt.axvline(stim, color='r')
        # plt.axvline(ts["t_start"][0], color='g')
        # for i in range(len(neg_starts)):
        #     plt.axvline(neg_starts[i], color='b')

        Phi = stim_start - ts["t_start"][0]
        Ti_0 = np.mean(neg_ends-neg_starts)
        Ti_0_std = np.std(neg_ends - neg_starts)
        T0 = np.mean(np.diff(neg_starts))
        T0_std = np.std(np.diff(neg_starts))
        T1 = ts["t_start"][1] - ts["t_start"][0]
        Theta = ts["t_start"][1] - stim_start
        Ti_1 = ts["t_end"][1] - ts["t_start"][1]
        Ti_2 = ts["t_end"][2] - ts["t_start"][2]
        if T0_std <= 150:
            res = (Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2)
            print(res)
            parameters_dict['data'].append(res)
            parameters_dict['count'] = parameters_dict['count'] + 1
            print(parameters_dict['count'])
        pickle.dump(parameters_dict, open(save_to, 'wb+'))
    return None

def ectract_data(data_folder):
    folders = get_folders(data_folder, "_prc")
    for folder in folders:
        suffixes = ['CH10']#['CH5', 'CH10', 'CH15']  #
        for suffix in suffixes:
            file = f'100_{suffix}_chunked.pkl'
            data = pickle.load(open(f'{data_folder}/{folder}/{file}', 'rb+'))
            save_to = f'../../data/parameters_prc_18032020_{folder}.pkl'
            extract_data_from_chunk(data, save_to)
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
        stim_start = chunk['stim_start']
        stim_end = chunk['stim_end']
        threshold = 0.1
        max_length = 50
        ts = plot_interact(PNA, stim_start, stim_end)
        if len(ts) == 8:
            ts1, te1, ts2, te2, ts3, te3, ts4, te4 = ts
            if (not np.isnan(ts1)) and (not np.isnan(te4)):
                Ti_0 = (te1 - ts1)
                T0 = (ts2 - ts1)
                Phi = (stim_start - ts2)
                Theta = (ts3 - stim_start)
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
    from numpy.polynomial.polynomial import Polynomial
    f = 10/3 # ms per one point
    data_ = pickle.load(open(file_load,'rb'))['data']
    data_ = np.array(data_)
    # data = get_rid_of_outliers(data)
    Phi = data_[:, 0] * f
    Ti_0 = data_[:, 1] * f
    T0 = data_[:, 2]* f
    T1 = data_[:, 3]* f
    Theta = data_[:, 4]* f
    Ti_1 = data_[:, 5]* f
    Ti_2 = data_[:, 6]* f
    phase = (Phi/np.mean(T0))
    cophase = (Theta/np.mean(T0))

    phi_insp = np.mean(Ti_0)/np.mean(T0)

    fig1 = plt.figure()
    plt.title("Phase-Cophase")
    y = cophase
    p = Polynomial.fit(phase, y,deg=6)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.axvline(phi_insp, color = 'k', linestyle='--')
    plt.grid(True)
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.xlabel("Phase")
    plt.ylabel("Cophase")
    plt.savefig(f"{dir_save_to}dataset_{num_rec}_phase_cophase.png")

    fig2 = plt.figure()
    plt.title("T1/T0")
    y = T1/T0
    p = Polynomial.fit(phase, y,deg=6)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.axvline(phi_insp, color = 'k', linestyle='--')
    plt.grid(True)
    # plt.xlim([0, 1])
    # plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    plt.ylim([0, 1.1 * np.max(y - np.mean(y)) + np.mean(y)])
    plt.xlabel("Phase")
    plt.savefig(f"{dir_save_to}dataset_{num_rec}_period_change.png")

    fig3 = plt.figure()
    y = T0
    p = Polynomial.fit(phase, y,deg=0)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.title("T0")
    plt.axvline(phi_insp, color = 'k', linestyle='--')
    plt.grid(True)
    # plt.xlim([0, 1])
    # plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    plt.ylim([0, 1500 * f])
    plt.ylabel("T0, ms")
    plt.xlabel("Phase")
    plt.savefig(open(f"{dir_save_to}dataset_{num_rec}_T0.png", "wb+"))

    fig4 = plt.figure()
    plt.title("T 1")
    y = T1
    p = Polynomial.fit(phase, y,deg=6)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.axvline(phi_insp, color = 'k', linestyle='--')
    # plt.xlim([0, 1])
    # plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    plt.ylim([0, 1500 * f])
    plt.ylabel("T1, ms")
    plt.xlabel("Phase")
    plt.savefig(f"{dir_save_to}dataset_{num_rec}_T1.png")

    fig5 = plt.figure()
    plt.title("Ti 0")
    y = Ti_0
    p = Polynomial.fit(phase, y,deg=0)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.axvline(phi_insp, color = 'k', linestyle='--')
    # plt.xlim([0, 1])
    # plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    plt.ylim([0, 300 * f])
    plt.ylabel("Ti 0, ms")
    plt.xlabel("Phase")
    plt.savefig(f"{dir_save_to}dataset_{num_rec}_Ti0.png")

    fig6 = plt.figure()
    plt.title("Ti 1")
    y = Ti_1
    p = Polynomial.fit(phase, y,deg=0)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.axvline(phi_insp, color = 'k', linestyle='--')
    # plt.xlim([0, 1])
    # plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    plt.ylabel("Ti 1, ms")
    plt.ylim([0, 300 * f])
    plt.xlabel("Phase")
    plt.savefig(f"{dir_save_to}dataset_{num_rec}_Ti1.png")

    fig7 = plt.figure()
    plt.title("Ti 2")
    y = Ti_2
    p = Polynomial.fit(phase, y,deg=0)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.axvline(phi_insp, color = 'k', linestyle='--')
    # plt.xlim([0, 1])
    # plt.ylim([1.1 * np.min(y-np.mean(y)) + np.mean(y), 1.1 * np.max(y-np.mean(y)) + np.mean(y)])
    plt.ylim([0, 300 * f])
    plt.ylabel("Ti 2, ms")
    plt.xlabel("Phase")
    plt.savefig(f"{dir_save_to}dataset_{num_rec}_Ti2.png")
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

if __name__ == '__main__':
    data_folder = '../../data/sln_prc'
    folder_save_to = '../../data/sln_prc_filtered'
    run_filtering(data_folder, folder_save_to)

    # split data into chunks
    data_folder = f'../../data/sln_prc_filtered'
    save_to = f'../../data/sln_prc_chunked'
    chunk_data(data_folder, save_to)

    data_folder = f'../../data/sln_prc_chunked'
    ectract_data(data_folder)

    # plotting final data
    for i in range(4):
        num_rec = i
        data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                      '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
        file_load = f'../../data/parameters_prc_18032020_{data_files[num_rec]}.pkl'
        dir_save_to = '../../img/experiments/'
        plot_final_data(data_files[num_rec], file_load, dir_save_to)

    # plotting parameter representation
    # num_rec = 3
    # num_chunk = 6
    # save_to = f'../../img/param_representation.png'
    # data = pickle.load(open(f'../../data/sln_prc_chunked/2019-09-05_12-26-14_prc/100_CH10_chunked.pkl', 'rb+'))
    # chunk = data[num_chunk]
    # clarifying_plot(chunk, save_to)
