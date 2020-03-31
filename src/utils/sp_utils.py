from copy import deepcopy
from scipy.signal import savgol_filter as sg
from sklearn.ensemble import IsolationForest as IF
from num_experiments.params_gen import *
from scipy import signal
from scipy.signal import butter
import ruptures as rpt
import peakutils

def detect_change_points(signal, model, pen, min_len):
    return rpt.Pelt(model=model, min_size=min_len).fit_predict(signal, pen)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def get_VNA_starts_and_ends(VNA, insp_starts, insp_ends):
    VNA = sg(VNA, 41, 3)
    VNA_reduced = deepcopy(VNA)
    min_val = np.quantile(VNA, 0.1)

    if insp_starts[0] > insp_ends[0]:
        insp_ends = insp_ends[1:]

    for i in range(np.minimum(len(insp_starts), len(insp_ends))):
        VNA_reduced[insp_starts[i] : insp_ends[i]] = min_val

    VNA_starts, VNA_ends = get_onsets_and_ends(VNA_reduced, model='l2', pen=100, min_len=100)
    return VNA_starts, VNA_ends


def get_insp_starts_and_ends(signal):
    signal_filtered = sg(signal, 121, 1)
    threshold = 0.4
    signal_binary = binarise_signal(signal_filtered, threshold)
    signal_change = np.diff(signal_binary)
    starts_inds = find_relevant_peaks(signal_change, 0.5, min_dist = 100)
    ends_inds = find_relevant_peaks(-signal_change, 0.5, min_dist = 100)
    return starts_inds,ends_inds

def get_period(signal):
    # for i in range(len(signals)):
    signal = signal - np.mean(signal)
    signal_filtered = sg(signal, 121, 1)
    threshold = np.quantile(signal_filtered, 0.8)
    signal_binary = binarise_signal(signal_filtered, threshold)
    # for correctness check
    # plt.plot(signal_filtered)
    # plt.plot(threshold * np.ones_like(signal_filtered))
    signal_change = np.diff(signal_binary)
    begins = find_relevant_peaks(signal_change, 0.5, min_dist = 100)
    ends = find_relevant_peaks(-signal_change, 0.5, min_dist = 100)
    T = np.median(np.hstack([np.diff(begins), np.diff(ends)]))
    std = np.std(np.hstack([np.diff(begins), np.diff(ends)]))
    #Ti =  ends - begins
    # Te - begins - ends
    return T, std

def binarise_signal(signal, threshold):
    res = np.zeros_like(signal)
    res[np.where(signal > threshold)] = 1.0
    res[np.where(signal <= threshold)] = 0
    return res

def last_lesser_than(alist, element):
    #sorted list
    if np.isnan(element):
        return [np.nan]
    for i in range(len(alist)):
        if alist[i] >= element:
            if i - 1 < 0:
                raise ValueError("The index can't be negative.")
            else:
                return alist[i - 1], i-1
    return [np.nan]

def first_greater_than(alist, element):
    if np.isnan(element):
        return [np.nan]
    for i in range(len(alist)):
        if alist[i] <= element:
            pass
        else:
            return alist[i], i
    return [np.nan]

def find_relevant_peaks(signal, threshold, min_dist):
    # peaks = scipy.signal.find_peaks(signal)[0]
    # peaks_filtered = peaks_above_threshold[np.where(np.diff(np.append(-np.inf, peaks_above_threshold)) > min_dist)]
    peaks = peakutils.indexes(signal, thres_abs=threshold, min_dist=min_dist)
    peaks_above_threshold = np.array([peaks[i] for i in range(len(peaks)) if abs(signal[peaks[i]]) > threshold])
    return peaks_above_threshold

def get_rid_of_outliers(data):
    cov = IF(contamination=0.25).fit(data)
    mask = (cov.fit_predict(data) + 1) // 2
    inds = np.nonzero(mask)
    data_filtered = data[inds,:].squeeze()
    return data_filtered

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

def get_onsets_and_ends(signal, model, pen, min_len):
    breakpoints = detect_change_points(signal, model, pen, min_len)
    # need to separete starts and ends
    window_len = 50
    mean_val = np.mean(signal)
    signal_ends = []
    signal_begins = []
    for t in breakpoints:
        ind_window_start = np.maximum(0, t - window_len)
        ind_window_end = np.minimum(len(signal) - 1, t + window_len)
        fall = np.mean(signal[ind_window_start : t]) > np.mean(signal[t : ind_window_end])
        if t == 0 or t == len(signal) - 1:
            pass
        if fall:
            signal_ends.append(t)
        else:
            signal_begins.append(t)
    return signal_begins, signal_ends

def get_number_of_breakthroughs(signal, min_amp):
    signal_filtered = sg(signal, 121, 1)[300:]
    threshold = (min_amp)
    signal_binary = binarise_signal(signal_filtered, threshold)
    # for correctness check
    # plt.plot(signal_filtered)
    # plt.plot(threshold * np.ones_like(signal_filtered))
    # identify gaps:
    signal_change = (np.diff(signal_binary))
    signal_begins = np.maximum(0, signal_change)
    signal_ends = np.minimum(0, signal_change)
    # get the indices of jumps
    signal_begins_inds = np.nonzero(signal_begins)[0]
    signal_ends_inds = np.nonzero(signal_ends)[0]
    num_breaks = (len(signal_begins_inds) + len(signal_ends_inds)) / 2
    return num_breaks



