import numpy as np
from num_experiments.run_model import run_model
from src.utils.sp_utils import *
import pickle
from tqdm.auto import tqdm
from copy import deepcopy
from num_experiments.params_gen import generate_params
import os
from utils.gen_utils import create_dir_if_not_exist, get_project_root


def get_features_from_signal(signal, dt, stim_start, stim_end):
    insp_begins, insp_ends = get_insp_starts_and_ends(signal)
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

# def get_features_short_impulse(signals, dt, t_stim_start, t_stim_finish):
#     #first one has to cut the relevant signal:
#     labels = ['PreI', 'EarlyI', "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2",
#               "Sw3", "KF_t", "KF_p", "KF_relay", "HN", "PN", "VN", "KF_inh", "NTS_inh"]
#     needed_labels = ["PreI"]
#     signals_relevant = [signals[i] for i in range(len(signals)) if labels[i] in needed_labels]
#     PreI = signals_relevant[needed_labels.index("PreI")]
#     PreI_filtered = savgol_filter(PreI, 121, 1)
#     threshold = 0.4 #np.quantile(PreI_filtered[20000: ], 0.65)
#     PreI_binary = binarise_signal(PreI_filtered, threshold)
#
#     #get the stimulation time_id
#     # stim_id = [peak_id for peak_id in scipy.signal.find_peaks(PostI)[0] if PostI[peak_id] > 0.5][0]
#     stim_id = int(t_stim_start / dt)
#
#     PreI_change = np.diff(PreI_binary)
#     PreI_begins = find_relevant_peaks(signal=PreI_change, threshold=0.5)
#     PreI_ends = find_relevant_peaks(signal=-1.0*PreI_change, threshold=0.5)
#
#     _, i = last_lesser_than(PreI_begins, stim_id)
#     begin_id = i - 1 # cause we need one more breathing cycle at the start
#     #some margin
#     starttime_id = PreI_begins[begin_id] - 500
#
#     stop_peak_id = i + 3
#     stoptime_id = PreI_ends[stop_peak_id] + 500
#
#     #discard unnessessary information
#     for i in range(len(signals_relevant)):
#         signals_relevant[i] = signals_relevant[i][starttime_id:stoptime_id]
#     PreI = signals_relevant[needed_labels.index("PreI")]
#     PreI_filtered = sg(PreI, 121, 1)
#     threshold = 0.4
#     PreI_binary = binarise_signal(PreI_filtered, threshold )
#     PreI_change = np.diff(PreI_binary)
#     PreI_begins = find_relevant_peaks(signal=PreI_change, threshold=0.5)
#     PreI_ends = find_relevant_peaks(signal=-PreI_change, threshold=0.5)
#     stim_id = stim_id - starttime_id
#
#     ts2 = last_lesser_than(PreI_begins, stim_id)[0]
#     ts3 = first_greater_than(PreI_begins, stim_id)[0]
#     ts1 = last_lesser_than(PreI_begins, ts2)[0]
#     ts4 = first_greater_than(PreI_begins, ts3)[0]
#
#     te1 = first_greater_than(PreI_ends, ts1)[0]
#     te2 = first_greater_than(PreI_ends, ts2)[0]
#     te3 = first_greater_than(PreI_ends, ts3)[0]
#     te4 = first_greater_than(PreI_ends, ts4)[0]
#
#     # plt.plot(PreI_filtered)
#     # plt.axvline(ts1, color='k')
#     # plt.axvline(ts2, color='r')
#     # plt.axvline(ts3, color='g')
#     # plt.axvline(ts4, color='b')
#     # plt.axvline(te1, color='k')
#     # plt.axvline(te2, color='r')
#     # plt.axvline(te3, color='g')
#     # plt.axvline(te4, color='b')
#     # plt.axvline(stim_id, color='m')
#     #identifying Ti_0, T0, T1, Phi, Theta (Phi + Theta + delta = T1), Ti_1, Ti_2:
#     Ti_0 = (te1 - ts1)*dt
#     T0 = (ts2-ts1)*dt
#     Phi = (stim_id - ts2) * dt
#     Theta = (ts3-stim_id) * dt
#     T1 = (ts3 - ts2) * dt
#     Ti_1 = (te3 - ts3) * dt
#     Ti_2 = (te4 - ts4) * dt
#     return Ti_0, T0, T1, Phi, Theta, Ti_1, Ti_2


def run_simulations(params, folder_save_to):
    generate_params(1, 1)
    # first, find the preiod, then create a list of points with the same phase if there are no stimulation at all
    stim_duration = params["stim_duration"] #250
    amp = params["amp"] #
    dt = params["dt"] #0.75
    stoptime = params["stoptime"] #70000
    num_shifts = params["num_shifts"] #100
    settle_time = params["settle_time"] #25000
    signals, t = run_model(dt, t_start=0, t_end=1, amp=0, stoptime=stoptime)
    # signals, t = pickle.load(open("../data/signals_intact_model.pkl", "rb+"))
    # get rid of transients 20000:
    # warning period is in indices not in ms!
    T, T_std = get_period(signals[0, settle_time:])
    # start from the end of expiration (begin of inspiration)
    PreI = signals[0, settle_time:]
    t_start_insp, t_end_insp = (get_insp_starts_and_ends(PreI))
    t_start_insp = (t_start_insp) * dt + settle_time
    t_end_insp = (t_end_insp) * dt + settle_time
    t1_s = t_start_insp[:5]
    # shifts in ms
    shifts = np.array([T * i / num_shifts for i in range(num_shifts)]) * dt

    for i in tqdm(range(len(shifts))[::-1]):
        for j in range(len(t1_s)):
            shift = shifts[i]
            t1 = int(t1_s[j] + shift)
            # print("Shift: {}, Impulse at time : {}".format(shift, t1))
            t2 = t1 + stim_duration
            # create and run a model
            signals, t = run_model(dt, t1, t2, amp, stoptime, folder_save_img_to)
            data = dict()
            data['signals'] = signals
            data['t'] = t
            data['dt'] = dt
            data['phase'] = np.round((2 * np.pi) * (i / len(shifts)), 2)
            data['shift'] = shift
            data['period'] = T
            data['period_std'] = T_std
            data['start_stim'] = t1
            data['duration'] = stim_duration
            data['amp'] = amp
            pickle.dump(data, open(f"{folder_save_to}/run_{amp}_{stim_duration}_{data['phase']}_{j}.pkl", "wb+"))
    return None

def extract_data(signals_path, save_to):
    file_names = os.listdir(signals_path)
    parameters_dict = {}
    parameters_dict['data'] = []
    for file_name in file_names:
        print(file_name)
        file = open(signals_path + "/" + file_name, "rb+")
        data = pickle.load(file)
        dt = data['dt']
        signals = data['signals']
        stim_start = int(data['start_stim']/dt)
        stim_duration = int(data['duration']/dt)
        stim_end = data['start_stim'] + stim_duration
        amp = data['amp']
        PreI = signals[0, :]
        res = get_features_from_signal(PreI, dt, stim_start, stim_end)
        parameters_dict['data'].append(deepcopy(res))
        file.close()
    pickle.dump(parameters_dict, open(f"{save_to}/info_var_phase_{amp}_{data['duration']}.pkl","wb+"))
    return None


if __name__ == '__main__':
    params = {}
    params["dt"] = 0.75
    params["stim_duration"] = 750
    stim_duration = params["stim_duration"]
    params["stoptime"] = 56000
    params["num_shifts"] = 50
    params["settle_time"] = 25000
    amps = [150, 250, 350]
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"
    save_extracted_data_to = data_path + '/' + "num_exp_results/short_stim/"
    for amp in amps:
        params["amp"] = amp
        folder_signals = f"{data_path}/num_exp_runs/short_stim/num_exp_short_stim_{amp}_{stim_duration}"
        create_dir_if_not_exist(folder_signals)
        run_simulations(params, folder_signals)
        extract_data(signals_path=folder_signals, save_to=save_extracted_data_to)





