import pickle
from matplotlib import pyplot as plt
from utils.gen_utils import get_project_root, get_folders
from scipy.signal import savgol_filter
from scipy.signal import hilbert
from tqdm.auto import tqdm
import numpy as np
from utils.sp_utils import butter_lowpass_filter, get_onsets_and_ends, get_timings, get_insp_starts_and_ends, scale
from utils.plot_utils import plot_power_spectrum
from copy import deepcopy
from scipy.optimize import minimize


def get_phase_shift(signal, dt, stim_start, stim_end, transient_offset):
    # if signal_source == 'simulations':
    #     pass
    # if signal_source == 'experiments':
    #     pass
    fs = 1000/dt #in Hz
    sig = savgol_filter(signal, 71, 3)
    sig = butter_lowpass_filter(sig, 1, fs, order=2)
    analytic_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    offset = np.mean(analytic_signal[:stim_start])
    shifted_analytic_signal = analytic_signal - offset
    instantaneous_phase = np.unwrap(np.angle(shifted_analytic_signal))
    instantaneous_phase_before = instantaneous_phase[:stim_start]
    instantaneous_phase_after = instantaneous_phase[stim_end + transient_offset:]
    amplitude_envelope_normalised = np.max(instantaneous_phase) * scale(amplitude_envelope)

    t_before = np.arange(len(instantaneous_phase_before))
    t_full = np.arange(len(instantaneous_phase))
    t_after = np.arange(len(instantaneous_phase))[stim_end + transient_offset:]

    # original phase is a * t + c (c=0)
    def fun_minimize1(a, t, y):
        return np.sum((a * t - y) ** 2)
    a = minimize(fun_minimize1, x0=np.random.rand(), args=(t_before, instantaneous_phase_before)).x
    # phase after all transients: a * x + b
    def fun_minimize2(b, a, t, y):
        return np.sum((a * t + b - y) ** 2)
    b = minimize(fun_minimize2, x0=np.random.rand(), args=(a, t_after, instantaneous_phase_after)).x[0]
    # old_phase = ax + c
    # new_phase = ax + b
    # delta phi = (c - b) / a
    delta_Phi = (- b) / a
    return delta_Phi[0]

def get_features_from_experimental_signal(signal, dt, stim_start, stim_end, transient_offset):
    insp_begins, insp_ends = get_onsets_and_ends(signal, model='l2', pen=1000, min_len=50)
    len_signal = len(signal)
    ts = get_timings(insp_begins, insp_ends, stim_start, len_signal)

    ind_neg_starts = np.where(np.array(list((ts["t_start"].keys()))).astype(int) < 0)[0]
    neg_starts = []
    for i in range(len(ind_neg_starts)):
        neg_starts.append(ts['t_start'][list(ts['t_start'].keys())[ind_neg_starts[i]]])
    neg_starts = np.array(neg_starts)[::-1]

    ind_neg_end = np.where(np.array(list((ts["t_end"].keys()))).astype(int) < 0)[0]
    neg_ends = []
    for i in range(len(ind_neg_end)):
        neg_ends.append(ts['t_end'][list(ts['t_end'].keys())[ind_neg_end[i]]])
    neg_ends = np.array(neg_ends)[::-1]

    Phi = (stim_start - ts["t_start"]["0"]) * dt
    Ti_0 = np.nanmean(neg_ends - neg_starts) * dt
    Ti_0_std = np.nanstd((neg_ends - neg_starts) * dt)
    T0 = np.nanmean(np.diff(neg_starts * dt))
    T0_std = np.nanstd(np.diff(neg_starts * dt))
    T1 = (ts["t_start"]["1"] - ts["t_start"]["0"]) * dt
    Theta = (ts["t_start"]["1"] - stim_start) * dt
    Ti_1 = (ts["t_end"]["1"] - ts["t_start"]["1"]) * dt
    Delta_Phi = get_phase_shift(signal, dt, stim_start, stim_end, transient_offset) * dt
    return Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_0_std, T0_std, Delta_Phi # all in ms

    # fig1 = plt.figure(figsize = (20,10))
    # plt.plot(t_full, amplitude_envelope_normalised, color = 'r', alpha = 0.2, linewidth = 3)
    # plt.plot(t_full, instantaneous_phase, color = 'k', linewidth = 3)
    # plt.plot(t_full, a * t_full + c, color = 'green', linewidth = 3)
    # plt.plot(t_after, a * t_after + b, color = 'blue', linewidth = 3)
    # plt.axvline(data_chunk["stim_start"] * 10/3, color = 'r')
    # plt.axvline(data_chunk["stim_end"] * 10/3, color = 'r')
    # plt.axvspan(data_chunk["stim_start"] * 10/3, data_chunk["stim_end"] * 10/3, color = 'r', alpha=0.2)
    # plt.xlabel("time, ms")
    # plt.legend(["amp_envelope", "proto phase", "hypothetical phase", "phase after reset"], fontsize = 24)
    # img_path = str(get_project_root()) + "/img"
    # fig1.savefig(f"{img_path}/other_plots/Hilbert_transform_real_recording.png")
    # plt.show(block=True)


def extract_PRC(dataset_chunks):
    # returns dictionary with extracted data: Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_0_std, T0_std, Delta_Phi
    cutoff_std = 150 # if the T has std more than cutoff_std * dt**2 then disregard the data
    transient_offset = 400 # data points after the application of the stim which correspond to transient dynamics
    list_chunks = list(dataset_chunks.keys())
    parameters_dict = {}
    parameters_dict['column_names'] = ["Phi", "Ti_0", "T0", "T1", "Theta", "Ti_1", "Ti_0_std", "T0_std", "Delta_Phi"]
    parameters_dict['data'] = []
    for chunk_num in tqdm(list_chunks):
        data_chunk = dataset_chunks[chunk_num]
        PNA = data_chunk['PNA']
        dt = dataset_chunks[chunk_num]['dt']
        stim_start = dataset_chunks[chunk_num]['stim_start']
        stim_end = dataset_chunks[chunk_num]['stim_end']
        Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_0_std, T0_std, Delta_Phi = get_features_from_experimental_signal(PNA, dt, stim_start, stim_end, transient_offset)
        if T0_std <= cutoff_std * dt**2:
            res = (Phi, Ti_0, T0, T1, Theta, Ti_1, Delta_Phi)
            parameters_dict['data'].append(res)
    parameters_dict['data'] = np.array(parameters_dict['data'])
    return parameters_dict

def run_PRC_extraction(data_folder, save_to):
    folders = get_folders(data_folder, "_prc")
    for folder in folders:
        file = f'chunked.pkl'
        data = pickle.load(open(f'{data_folder}/{folder}/{file}', 'rb+'))
        params_save_to = f'{save_to}/parameters_prc_{folder}.pkl'
        params_extracted = extract_PRC(data)
        pickle.dump(params_extracted, open(params_save_to, 'wb+'))
    return None

if __name__ == '__main__':
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"
    data_folder = f'{data_path}/sln_prc_chunked'
    save_to = f'{data_path}/sln_prc_params'
    run_PRC_extraction(data_folder, save_to)
