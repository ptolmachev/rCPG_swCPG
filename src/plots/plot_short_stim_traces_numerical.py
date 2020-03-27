### plotting numerical experimental traces
import pickle
from utils.gen_utils import create_dir_if_not_exist, get_files, get_project_root
from utils.plot_utils import plot_num_exp_traces

params = {}
amps = [150, 250, 350]
stim_duration = 500
data_path = str(get_project_root()) + "/data"
img_path = str(get_project_root()) + "/img"
for amp in amps:
    params["amp"] = amp
    folder_signals = f"num_exp_runs/num_exp_short_stim_{amp}_{stim_duration}"
    folder_save_img_to = f"traces"
    create_dir_if_not_exist(img_path + "/" + f"num_experiments/short_stim/short_stim_{amp}_{stim_duration}/{folder_save_img_to}")
    files = get_files(root_folder=f"../data/num_exp_runs/num_exp_short_stim_{amp}_{stim_duration}", pattern=".pkl")
    for file in files:
        data = pickle.load(open(data_path +"/" +folder_signals + "/" + file, "rb+"))
        save_to = ""
        signals = data['signals']
        name = file.split(".pkl")[0] + ".png"
        plot_num_exp_traces(signals, img_path + "/" + f"num_experiments/short_stim/short_stim_{amp}_{stim_duration}/traces/{name}")