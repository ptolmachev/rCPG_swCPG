import numpy as np
from num_experiments.run_model import run_model
from src.utils.sp_utils import *
import pickle
from tqdm.auto import tqdm
from copy import deepcopy
from num_experiments.params_gen import generate_params
import os
from utils.gen_utils import create_dir_if_not_exist, get_project_root


# def get_phase_shift(signal, dt, stim_start, stim_end, transient_offset):
#     fs = 1000/dt #in Hz
#     sig = savgol_filter(signal, 71, 3)
#     sig = butter_lowpass_filter(sig, 1, fs, order=2)
#     analytic_signal = hilbert(sig)
#     amplitude_envelope = np.abs(analytic_signal)
#     offset = np.mean(analytic_signal[:stim_start])
#     shifted_analytic_signal = analytic_signal - offset
#     instantaneous_phase = np.unwrap(np.angle(shifted_analytic_signal))
#     instantaneous_phase_before = instantaneous_phase[:stim_start]
#     instantaneous_phase_after = instantaneous_phase[stim_end + transient_offset:]
#     amplitude_envelope_normalised = np.max(instantaneous_phase) * scale(amplitude_envelope)
#
#     t_before = np.arange(len(instantaneous_phase_before))
#     t_full = np.arange(len(instantaneous_phase))
#     t_after = np.arange(len(instantaneous_phase))[stim_end + transient_offset:]
#
#     # original phase is a * t + c (c=0)
#     def fun_minimize1(a, t, y):
#         return np.sum((a * t - y) ** 2)
#     a = minimize(fun_minimize1, x0=np.random.rand(), args=(t_before, instantaneous_phase_before)).x
#     # phase after all transients: a * x + b
#     def fun_minimize2(b, a, t, y):
#         return np.sum((a * t + b - y) ** 2)
#     b = minimize(fun_minimize2, x0=np.random.rand(), args=(a, t_after, instantaneous_phase_after)).x[0]
#     # old_phase = ax + c
#     # new_phase = ax + b
#     # delta phi = (c - b) / a
#     delta_Phi = (- b) / a
#     return delta_Phi[0]
from utils.plot_utils import plot_num_exp_traces


def get_features_from_numerical_signal(signal, dt, stim_start_ind, stim_end_ind):
    settle_time_inds = 15000
    insp_begins_, insp_ends_ = get_insp_starts_and_ends(signal)
    # discard transients
    insp_begins = []
    insp_ends = []
    for i in range(len(insp_begins_)):
        if insp_begins_[i] > settle_time_inds:
            insp_begins.append(insp_begins_[i])
    for i in range(len(insp_ends_)):
        if insp_ends_[i] > settle_time_inds:
            insp_ends.append(insp_ends_[i])

    len_signal = len(signal)
    # TODO define period
    ts = get_timings(insp_begins, insp_ends, stim_start_ind, len_signal)
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

    Phi = (stim_start_ind - ts["t_start"]["0"]) * dt
    Ti_0 = np.mean(neg_ends - neg_starts) * dt
    Ti_0_std = np.std((neg_ends - neg_starts) * dt)
    T0 = np.mean(np.diff(neg_starts * dt))
    T0_std = np.std(np.diff(neg_starts * dt))
    T1 = (ts["t_start"]["1"] - ts["t_start"]["0"]) * dt
    Theta = (ts["t_start"]["1"] - stim_start_ind) * dt
    Ti_1 = (ts["t_end"]["1"] - ts["t_start"]["1"]) * dt
    return Phi, Ti_0, T0, T1, Theta, Ti_1, Ti_0_std, T0_std # all in ms

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
            signals, t = run_model(dt, t1, t2, amp, stoptime)
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
        stim_start_ind = int(data['start_stim']/dt)
        stim_duration = int(data['duration']/dt)
        stim_end_ind = stim_start_ind + int(stim_duration/dt)
        amp = data['amp']
        PreI = signals[0, :]
        res = get_features_from_numerical_signal(PreI, dt, stim_start_ind, stim_end_ind)
        parameters_dict['data'].append(deepcopy(res))
        file.close()
    pickle.dump(parameters_dict, open(f"{save_to}/info_var_phase_{amp}_{data['duration']}.pkl","wb+"))
    return None


if __name__ == '__main__':
    params = {}
    params["dt"] = 0.75
    params["stoptime"] = 65000
    params["num_shifts"] = 50
    params["settle_time"] = 25000
    amps = [150]
    stim_durations = [500]
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"
    save_extracted_data_to = data_path + '/' + "num_exp_results/short_stim/"
    for stim_duration in stim_durations:
        for amp in amps:
            params["amp"] = amp
            params["stim_duration"] = stim_duration
            print(amp, stim_duration)
            folder_signals = f"{data_path}/num_exp_runs/short_stim/num_exp_short_stim_{amp}_{stim_duration}"
            # create_dir_if_not_exist(folder_signals)
            # run_simulations(params, folder_signals)
            extract_data(signals_path=folder_signals, save_to=save_extracted_data_to)