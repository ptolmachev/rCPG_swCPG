### PLOTTING LONG STIM DATA
import pickle
from copy import deepcopy
import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from utils.gen_utils import get_folders, get_project_root
from utils.plot_utils import plot_chunk
from utils.sp_utils import find_relevant_peaks
from tqdm.auto import tqdm

data_path = str(get_project_root()) + "/data"
img_path = str(get_project_root()) + "/img"

data_folders = get_folders(f"{data_path}/sln_prc_filtered/", "_t")
for folder in tqdm(data_folders):
    HNA = pickle.load(open(f'{data_path}/sln_prc_filtered/{folder}/100_CH5_processed.pkl', 'rb+'))
    PNA = pickle.load(open(f'{data_path}/sln_prc_filtered/{folder}/100_CH10_processed.pkl', 'rb+'))
    VNA = pickle.load(open(f'{data_path}/sln_prc_filtered/{folder}/100_CH15_processed.pkl', 'rb+'))
    data = {}
    stim_duration = int(PNA['stim_end'] - PNA['stim_start'])
    signal_start = int(np.maximum(0, PNA['stim_start'] - stim_duration))
    signal_end =  int(np.minimum(PNA['stim_end'] + stim_duration, len(PNA['signal']) - 1))
    data["HNA"] = deepcopy(HNA['signal'][signal_start:signal_end])
    data["PNA"] = deepcopy(PNA['signal'][signal_start:signal_end])
    data["VNA"] = deepcopy(VNA['signal'][signal_start:signal_end])
    data["stim_start"] = int(PNA['stim_start'] - signal_start)
    data["stim_end"] = int(PNA['stim_end'] - signal_start)
    VNA_stim_response = savgol_filter(data["VNA"][data["stim_start"]:data["stim_end"]], 51, 1)
    HNA_stim_response = savgol_filter(data["HNA"][data["stim_start"]:data["stim_end"]], 51, 1)
    thr_HNA = np.quantile(HNA_stim_response, 0.65)
    thr_VNA = np.quantile(VNA_stim_response, 0.65)
    peak_HNA_inds = find_relevant_peaks(HNA_stim_response, threshold=thr_HNA, min_dist=150)
    peak_VNA_inds = find_relevant_peaks(VNA_stim_response, threshold=thr_VNA, min_dist=150)

    fig = plot_chunk(data, y_lim = [1, 18])
    for i in range(len(fig.axes)):
        for j in range(len(peak_HNA_inds)):
            fig.axes[i].axvline(peak_HNA_inds[j] + stim_duration, color = 'purple')
        for j in range(len(peak_VNA_inds)):
            fig.axes[i].axvline(peak_VNA_inds[j] + stim_duration, color = 'green')
    fig.savefig(f"{img_path}/experiments/traces/long_stim/{folder}")
    plt.close(fig)
