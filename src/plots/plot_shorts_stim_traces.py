import pickle
from matplotlib import pyplot as plt
from utils.gen_utils import get_project_root, get_folders
from utils.plot_utils import plot_chunk
from tqdm.auto import tqdm

### PLOTTING CHUNKS
data_path = str(get_project_root()) + "/data"
img_path = str(get_project_root()) + "/img"
# data_folders = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc', '2019-09-05_12-26-14_prc']
data_folders = get_folders(f"{data_path}/sln_prc_chunked/", "_prc")
for i in (range(len(data_folders))):
    file_load = f'{data_path}/sln_prc_chunked/{data_folders[i]}/chunked.pkl'
    data = pickle.load(open(file_load, 'rb+'))
    list_chunks = list(data.keys())
    for chunk_num in tqdm(list_chunks):
        data_chunk = data[chunk_num]
        fig = plot_chunk(data_chunk, y_lim=[1, 18])
        fig.savefig(f"{img_path}/experiments/traces/short_stim/{i}_{chunk_num}")
        plt.close(fig)