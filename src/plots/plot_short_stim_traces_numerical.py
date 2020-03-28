### plotting numerical experimental traces
import pickle
from utils.gen_utils import create_dir_if_not_exist, get_files, get_project_root
from utils.plot_utils import plot_num_exp_traces
from tqdm.auto import tqdm

params = {}
amps = [150, 250, 350]
stim_duration = 500
data_path = str(get_project_root()) + "/data"
img_path = str(get_project_root()) + "/img"
for amp in tqdm(amps):
    params["amp"] = amp
    folder_signals = f"{data_path}/num_exp_runs/short_stim/num_exp_short_stim_{amp}_{stim_duration}"
    folder_save_img_to = img_path + "/" + f"num_experiments/short_stim/short_stim_{amp}_{stim_duration}/traces"
    create_dir_if_not_exist(folder_save_img_to)
    files = get_files(root_folder=folder_signals, pattern=".pkl")
    for file in files:
        data = pickle.load(open(folder_signals + "/" + file, "rb+"))
        save_to = ""
        signals = data['signals']
        name = file.split(".pkl")[0] + ".png"
        plot_num_exp_traces(signals, folder_save_img_to)