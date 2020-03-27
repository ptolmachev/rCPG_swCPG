import pickle
from utils.gen_utils import get_folders, get_project_root, create_dir_if_not_exist
from utils.sp_utils import get_period
from tqdm.auto import tqdm


def split_signal_into_chunks(signals, stim_starts, stim_end):
    # for span between the two stims
    PNA, VNA, HNA = signals
    data_chunked = dict()
    for i in tqdm(range(len(stim_starts) - 1)):
        # print(f'chunk number {i}')
        start = stim_starts[i]
        end = stim_starts[i + 1]
        chunk = PNA[int(start):int(end)]
        # find period
        T, std_T = get_period(chunk)
        new_chunk_PNA = PNA[int(start + int(1 * T)):int(end + int(4 * T))]
        new_chunk_VNA = VNA[int(start + int(1 * T)):int(end + int(4 * T))]
        new_chunk_HNA = HNA[int(start + int(1 * T)):int(end + int(4 * T))]
        data_chunked[i] = dict()
        data_chunked[i]['PNA'] = new_chunk_PNA
        data_chunked[i]['VNA'] = new_chunk_VNA
        data_chunked[i]['HNA'] = new_chunk_HNA
        data_chunked[i]['T'] = T
        data_chunked[i]['std_T'] = std_T
        data_chunked[i]['stim_start'] = new_chunk_PNA.shape[-1] - int(4 * T)
        data_chunked[i]['stim_end'] = new_chunk_PNA.shape[-1] + int((stim_end[i] - stim_starts[i])) - int(4 * T)
    return data_chunked

def chunk_data(data_folder, save_to):
    '''splits huge recording into chunks with one stimulus per chunk'''
    folders = get_folders(data_folder, "_prc")
    for folder in tqdm(folders):

        VNA_data = pickle.load(open(f'{data_folder}/{folder}/100_CH15_processed.pkl', 'rb+'))
        VNA = VNA_data['signal']

        HNA_data = pickle.load(open(f'{data_folder}/{folder}/100_CH5_processed.pkl', 'rb+'))
        HNA = HNA_data['signal']

        PNA_data = pickle.load(open(f'{data_folder}/{folder}/100_CH10_processed.pkl', 'rb+'))
        PNA = PNA_data['signal']

        stim_start = PNA_data['stim_start']
        stim_end = PNA_data['stim_end']
        signals = [PNA, VNA, HNA]

        data_chunked = split_signal_into_chunks(signals, stim_start, stim_end)
        create_dir_if_not_exist(f'{save_to}/{folder}')
        pickle.dump(data_chunked, open(f'{save_to}/{folder}/chunked.pkl', 'wb+'))
    return None


if __name__ == '__main__':
    # split data into chunks
    data_path = str(get_project_root()) + "/data"
    data_folder = f'{data_path}/sln_prc_filtered'
    save_to = f'{data_path}/sln_prc_chunked'
    create_dir_if_not_exist(save_to)
    chunk_data(data_folder, save_to)