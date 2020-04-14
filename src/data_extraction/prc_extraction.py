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
from scipy.optimize import minimize, curve_fit, leastsq
from scipy.integrate import quad
import sympy


def function_to_fit(x, th, order):
    res = x[0] * np.ones_like(th)
    for i in range(order):
        res += x[1 + i] * np.cos((i+1) * th) + x[1 + i + order] * np.sin((i+1) * th)
    return res

def constr_fun(x):
    return x[0] - 1/(2*np.pi)

def func_to_minimise(x, th, y, order):
    return np.sum((function_to_fit(x, th, order) - y) ** 2)

def fit_sigma(th, y, order):
    cons = {'type': 'eq', 'fun': constr_fun}
    res = minimize(func_to_minimise, x0=np.random.rand(2*order + 1), args=(th, y, order), constraints=cons)
    if res.success == False:
        print("The fit wasn't successful")
        return None
    else:
        return res.x

def get_protophase_to_phase_mapping(protophase, omega):
    # protophase_dot = np.diff(protophase)
    # x = (protophase)[1:] % (2 * np.pi)
    # y = (1 / protophase_dot) * omega
    n_bins = 200
    transient_inds = 100
    res = np.histogram(protophase[transient_inds:] % (2 * np.pi), bins=n_bins, range=[0, 2 * np.pi], density=True)
    th = (res[1] - 2 * np.pi / (n_bins * 2))[1:]
    y = res[0]
    # to neglect transients
    # th = th[150:-150]
    # y = y[150:-150]
    # Do the parameter fitting
    order = 25
    coeff = fit_sigma(th, y, order)

    # figure = plt.figure(figsize=(20, 10))
    # plt.scatter(th, y, s=4)
    # plt.plot(np.sort(th), function_to_fit(coeff, np.sort(th), order=order), linewidth=3, color='r')
    # plt.grid(True)
    # # plt.ylim([-1, 5])
    # plt.show(block = True)

    if not (coeff is None):
        z = sympy.Symbol('z')
        expr = coeff[0]* 2*np.pi
        for i in range(order):
            expr += (coeff[i + 1] * sympy.cos((i + 1) * z) + coeff[i + 1 + order] * sympy.sin((i + 1) * z)) * 2*np.pi
        integrated_sigma = sympy.lambdify(z, sympy.integrate(expr, (z, 0, z)), 'numpy')
        return integrated_sigma
    else:
        return None

def get_phase_shift(signl, dt, stim_start, stim_end, transient_offset):
    fs = 1000/dt #in Hz
    signl_filtered = butter_lowpass_filter(signl, 0.7, fs, order=2)

    # first, figure out the frequency of oscillations (in cycles per index)
    # _b - before the stimulus, _a - after
    signl_b = signl_filtered[:stim_start]
    analytic_signal_b = hilbert(signl_b)
    offset = np.mean(analytic_signal_b)
    shifted_analytic_signal_b = analytic_signal_b - offset
    protophase = np.unwrap(np.angle(hilbert(signl_filtered) - offset))
    protophase_b = protophase[: stim_start ]
    protophase_a = protophase[stim_start + transient_offset:]

    # original phase is omega * t + c
    def line(x, t, y):
        omega, c = x
        return np.sum((omega * t + c - y) ** 2)

    omega, c = minimize(line, x0=np.random.rand(2), args=(np.arange(len(protophase_b)), protophase_b)).x
    #second, define the mapping from protophase to phase
    protophase_to_phase = get_protophase_to_phase_mapping(protophase_b, omega)
    if protophase_to_phase is None:
        return np.nan

    # transient_inds = 100
    # fig = plt.figure(figsize=(20, 10))
    # res = np.histogram(protophase_b[transient_inds:] % (2 * np.pi), bins=100, range=[0, 2 * np.pi], density=True)
    # plt.plot(res[1][:-1], res[0], linewidth=2, label="histogram protophase")
    # res = np.histogram(protophase_to_phase(protophase_b[transient_inds:]) % (2 * np.pi), bins=100, range=[0, 2 * np.pi], density=True)
    # plt.plot(res[1][:-1], res[0], linewidth=2, label="histogram phase")
    # plt.grid(True)
    # plt.xlabel("Phase", fontsize=24)
    # plt.ylabel("Density", fontsize=24)
    # plt.legend(fontsize=24)
    # plt.show(block=True)

    analytic_signal = hilbert(signl_filtered)
    shifted_analytic_signal = analytic_signal - offset
    phase = protophase_to_phase(protophase)
    # having the phase, find the phase shift
    transient_offset = 300
    phase_b = phase[:stim_start]
    t_b = np.arange(len(phase_b))
    phase_a = phase[stim_start + transient_offset:]
    t_a = np.arange(len(phase))[stim_start + transient_offset:]

    def constr_fun(x):
        return x[0] - omega

    a, c = minimize(line, x0=np.random.rand(2), args=(t_b, phase_b)).x
    a, b = minimize(line, x0=np.random.rand(2), args=(t_a, phase_a), constraints={'type': 'eq', 'fun': constr_fun}).x
    delta_Phi = (c - b) / a

    # plot to check
    # fig = plt.figure(figsize=(20, 5))
    # plt.plot(phase - (a * np.arange(len(phase)) + c), color='r', linewidth=3)
    # plt.plot(protophase - (a * np.arange(len(protophase)) + c), color='b', linewidth=3)
    # # t = np.arange(len(phase))
    # # plt.plot(t[:stim_start], a * t[:stim_start] + c, color='b')
    # # plt.plot(t[stim_start:], a * t[stim_start:] + b, color='g')
    # # plt.plot(t, signl, color='r', alpha=0.7)
    # plt.axvline(stim_start, color='k')
    # plt.axvline(stim_end, color='k')
    # plt.grid(True)
    # plt.ylabel("Deviation from the straight line", fontsize=24)
    # plt.show(block=True)
    # plt.close()
    # fig = plt.figure(figsize=(20, 5))
    # plt.plot(phase, color='r', linewidth=3)
    # plt.plot(protophase , color='b', linewidth=3)
    # plt.show(block=True)
    # plt.close()
    return delta_Phi #in inds not in ms

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
    for i, folder in enumerate(folders):
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
