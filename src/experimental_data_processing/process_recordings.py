import pickle
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np
from scipy.signal import decimate, convolve
from utils.gen_utils import get_folders, create_dir_if_not_exist, get_project_root
from utils.openphys_utils import load
from utils.sp_utils import butter_bandpass_filter


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
        filtered_data[t1:t2] = (filtered_data[t1] + filtered_data[t2])/2.0
    return filtered_data

def run_filtering_short_stim(data_folder, folder_save_to):
    folders = get_folders(data_folder, "prc")
    suffixes = ['CH5', 'CH10', 'CH15']
    fr = 30000
    cutoff_fr_high = 3000
    cutoff_fr_low = 300
    stim_spike_threshold = -4.95
    offset=100
    smoothing_window=1000
    sigma=150
    downsampling_factor=10
    for folder in tqdm(folders):
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
            create_dir_if_not_exist(f'{folder_save_to}/{folder}')
            pickle.dump(data_to_save, open(f'{folder_save_to}/{folder}/100_{suffix}_processed.pkl', 'wb+'))
    return None


def run_filtering_long_stim(data_folder, folder_save_to):
    folders = get_folders(data_folder, "_t")
    suffixes = ['CH5', 'CH10', 'CH15']
    fr = 30000
    cutoff_fr_high = 3000
    cutoff_fr_low = 300
    stim_spike_threshold = -4.95
    offset=100
    smoothing_window=1000
    sigma=150
    downsampling_factor=10
    for folder in tqdm(folders):
        stim = load(f'{data_folder}/{folder}/100_ADC1.continuous', dtype=float)["data"]
        stim_spikes_start, stim_spikes_end = find_stim_spikes(stim, threshold = stim_spike_threshold)
        stim_start = stim_spikes_start[0] / downsampling_factor ** 2
        stim_end = stim_spikes_end[-1] / downsampling_factor ** 2
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
            create_dir_if_not_exist(f'{folder_save_to}/{folder}')
            pickle.dump(data_to_save, open(f'{folder_save_to}/{folder}/100_{suffix}_processed.pkl', 'wb+'))
    return None

if __name__ == '__main__':
    data_path = str(get_project_root()) + "/data"
    # # FILTERING SHORT STIM DATA
    data_folder = f'{data_path}/sln_prc'
    folder_save_to = f'{data_path}/sln_prc_filtered'
    create_dir_if_not_exist(folder_save_to)
    run_filtering_short_stim(data_folder, folder_save_to)

    # FILTERING LONG STIM DATA
    # data_folder = f'{data_path}/sln_prc'
    # folder_save_to = f'{data_path}/sln_prc_filtered'
    # create_dir_if_not_exist(folder_save_to)
    # run_filtering_long_stim(data_folder, folder_save_to)