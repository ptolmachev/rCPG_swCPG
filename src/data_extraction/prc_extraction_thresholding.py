import pickle
from matplotlib import pyplot as plt
from utils.gen_utils import get_project_root, get_folders, create_dir_if_not_exist
from scipy.signal import savgol_filter
from scipy.signal import hilbert
from tqdm.auto import tqdm
import numpy as np
from utils.sp_utils import butter_lowpass_filter, get_onsets_and_ends, get_timings, get_insp_starts_and_ends, scale
from copy import deepcopy
from numpy.polynomial.polynomial import Polynomial

def get_phase_shift(signal, dt, stim_start, stim_end, params):
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
    Delta_Phi = 2 * np.pi * (T1 - T0) / (T0)
    Phi = 2 * np.pi * (Phi / T0)
    return Phi, Delta_Phi


def extract_PRC(dataset_chunks):
    list_chunks = list(dataset_chunks.keys())
    data = []

    for chunk_num in tqdm(list_chunks):
        data_chunk = dataset_chunks[chunk_num]
        PNA = data_chunk['PNA']
        dt = dataset_chunks[chunk_num]['dt']
        stim_start = dataset_chunks[chunk_num]['stim_start']
        stim_end = dataset_chunks[chunk_num]['stim_end']
        params = {}
        Phi, Delta_Phi = get_phase_shift(PNA, dt, stim_start, stim_end, params)
        data.append((Phi, Delta_Phi))
    return np.array(data)


if __name__ == '__main__':
    method = 'prc_threshold'
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"

    data_folder = f'{data_path}/sln_prc_chunked'
    folders = get_folders(data_folder, "_prc")

    # # Prepare data
    # for i, folder in enumerate(folders):
    #     file = f'chunked.pkl'
    #     dataset_chunks = pickle.load(open(f'{data_folder}/{folder}/{file}', 'rb+'))
    #     data = extract_PRC(dataset_chunks)
    #     save_to = f'{data_path}/exp_results_phase_shift/{method}/temp_data/{folder}'# /data.pkl'
    #     create_dir_if_not_exist(save_to)
    #     pickle.dump(data, open(save_to + f"/data.pkl", 'wb+'))

    # plot data
    ind_datasets = [0, 1,2,3]
    data = []
    data_folder = f'{data_path}/exp_results_phase_shift/{method}/temp_data/'
    folders = get_folders(data_folder, "_prc")
    for i, folder in enumerate(folders):
        if i in ind_datasets:
            load_from = f'{data_path}/exp_results_phase_shift/{method}/temp_data/{folder}'
            data.append(pickle.load(open(load_from + f"/data.pkl", 'rb+')))
    data = np.vstack(data)
    data = data[data[:, 0].argsort(), :]

    Phi = data[:, 0]
    Delta_Phi = data[:, 1]
    # poly = Polynomial.fit(Phi, Delta_Phi, deg=10)
    plt.scatter(Phi, Delta_Phi)
    # plt.plot(Phi, poly(Phi), color='r')

    plt.grid(True)
    plt.show(block=True)