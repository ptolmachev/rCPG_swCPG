import pickle
import numpy as np
from utils.gen_utils import get_project_root, get_folders
from utils.sp_utils import get_onsets_and_ends, get_timings


def get_features_from_signal(signal, dt, stim_start, stim_end):
    insp_begins, insp_ends = get_onsets_and_ends(signal, model='l2', pen=1000, min_len=60)
    len_signal = len(signal)
    ts = get_timings(insp_begins, insp_ends, stim_start, len_signal)

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

    Phi = (stim_start - ts["t_start"][0]) * dt
    Ti_0 = np.mean(neg_ends - neg_starts) * dt
    Ti_0_std = np.std((neg_ends - neg_starts) * dt)
    T0 = np.mean(np.diff(neg_starts * dt))
    T0_std = np.std(np.diff(neg_starts * dt))
    T1 = (ts["t_start"][1] - ts["t_start"][0]) * dt
    Theta = (ts["t_start"][1] - stim_start) * dt
    Ti_1 = (ts["t_end"][1] - ts["t_start"][1]) * dt
    Ti_2 = (ts["t_end"][2] - ts["t_start"][2]) * dt
    return Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2, Ti_0_std, T0_std # all in ms

def extract_data_from_chunks(dataset_chunks, dt, save_to):
    parameters_dict = {}
    parameters_dict['data'] = []
    print(f"The number of chunks of keys: {len(list(dataset_chunks.keys()))}")
    for num in list(dataset_chunks.keys()):
        print(f"chunk number: {num}")
        chunk = dataset_chunks[num]
        PNA = chunk['PNA']
        stim_start = chunk['stim_start']
        stim_end = chunk['stim_end']
        Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2, Ti_0_std, T0_std = get_features_from_signal(PNA, dt, stim_start, stim_end)
        if T0_std <= 150 * dt**2:
            res = (Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_2)
            print(res)
            parameters_dict['data'].append(res)
        pickle.dump(parameters_dict, open(save_to, 'wb+'))
    return None

def ectract_data(data_folder, dt):
    folders = get_folders(data_folder, "_prc")
    for folder in folders:
        file = f'chunked.pkl'
        data = pickle.load(open(f'{data_folder}/{folder}/{file}', 'rb+'))
        save_to = f'../../data/parameters_prc_{folder}.pkl'
        extract_data_from_chunks(data, dt, save_to)
    return None

if __name__ == '__main__':
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"

    dt = 10/3
    data_folder = f'{data_path}/sln_prc_chunked'
    ectract_data(data_folder, dt)

