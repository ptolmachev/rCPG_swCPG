import pickle
from matplotlib import pyplot as plt
from utils.gen_utils import get_project_root, get_folders, create_dir_if_not_exist
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
from numpy.polynomial.polynomial import Polynomial

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

# original phase is omega * t + c
def line(x, t, y):
    omega, c = x
    return np.sum((omega * t + c - y) ** 2)

def map_protophasse_to_phase(protophase, n_bins, transient_inds, order):
    # protophase_dot = np.diff(protophase)
    # x = (protophase)[1:] % (2 * np.pi)
    # y = (1 / protophase_dot) * omega
    res = np.histogram(protophase[transient_inds:] % (2 * np.pi), bins=n_bins, range=[0, 2 * np.pi], density=True)
    th = (res[1] - 2 * np.pi / (n_bins * 2))[1:]
    y = res[0]
    coeff = fit_sigma(th, y, order)
    if not (coeff is None):
        z = sympy.Symbol('z')
        expr = coeff[0]* 2*np.pi
        for i in range(order):
            expr += (coeff[i + 1] * sympy.cos((i + 1) * z) + coeff[i + 1 + order] * sympy.sin((i + 1) * z)) * 2*np.pi
        integrated_sigma = sympy.lambdify(z, sympy.integrate(expr, (z, 0, z)), 'numpy')
        return integrated_sigma
    else:
        return None

def get_phase_shift(signl, dt, stim_start, stim_end, params):
    transient_offset = params['transient_offset']
    signl_filtered = butter_lowpass_filter(signl, 0.0023, 1, order=2)
    # first, figure out the frequency of oscillations (in cycles per index)
    # _b - before the stimulus, _a - after
    signl_b = signl_filtered[:stim_start]
    analytic_signal_b = hilbert(signl_b)
    offset = np.mean(analytic_signal_b)
    protophase = np.unwrap(np.angle(hilbert(signl_filtered) - offset))
    protophase_b = protophase[: stim_start ]

    #second, define the mapping from protophase to phase
    protophase_to_phase = map_protophasse_to_phase(protophase_b, n_bins=200, transient_inds=100, order=50)
    if protophase_to_phase is None:
        return np.nan

    phase = protophase_to_phase(protophase)
    # having the phase, find the phase shift
    phase_b = phase[:stim_start]
    t_b = np.arange(len(phase_b)) * dt
    phase_a = phase[stim_start + transient_offset:]
    t_a = np.arange(len(phase))[stim_start + transient_offset:] * dt
    omega, c = minimize(line, x0=np.random.rand(2), args=(t_b, phase_b)).x
    def constr_fun(x):
        return x[0] - omega
    omega, b = minimize(line, x0=np.random.rand(2), args=(t_a, phase_a), constraints={'type': 'eq', 'fun': constr_fun}).x
    Delta_Phi = (c - b)
    Phi = phase[stim_start] % (2 * np.pi)
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
        params = {"transient_offset" : 500}
        Phi, Delta_Phi = get_phase_shift(PNA, dt, stim_start, stim_end, params)
        data.append((Phi, Delta_Phi))
    return np.array(data)


if __name__ == '__main__':
    method = 'prc_linear_fit'
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"

    data_folder = f'{data_path}/sln_prc_chunked'
    folders = get_folders(data_folder, "_prc")

    # # Prepare data
    # for i, folder in enumerate(folders):
    #     file = f'chunked.pkl'
    #     dataset_chunks = pickle.load(open(f'{data_folder}/{folder}/{file}', 'rb+'))
    #     data = extract_PRC(dataset_chunks)
    #     save_to = f'{data_path}/exp_results_phase_shift/{method}/temp_data/{folder}'
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
    poly = Polynomial.fit(Phi, Delta_Phi, deg=8)
    plt.scatter(Phi, Delta_Phi)
    plt.plot(Phi, poly(Phi), color='r')

    plt.grid(True)
    plt.show(block=True)
