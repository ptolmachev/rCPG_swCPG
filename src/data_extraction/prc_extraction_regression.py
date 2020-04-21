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
from tqdm.auto import tqdm
from numpy.linalg import lstsq

# some structure which stores relevant phase window (after the stimulus) and the function d phi /dt - omega in this window.
# algorithm
# load data one by one
# estimate its phase (time series)
# save the phase window after the stimulus alongside with the (d phi/dt - omega) function and a window function П(phi)

def line(x, t, y):
    omega, c = x
    return np.sum((omega * t + c - y) ** 2)

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

def get_phase(signl, stim_start):
    signl_filtered = butter_lowpass_filter(signl, 0.0025, 1, order=2)
    # first, figure out the frequency of oscillations (in cycles per index)
    # _b - before the stimulus, _a - after
    signl_b = signl_filtered[:stim_start]
    analytic_signal_b = hilbert(signl_b)
    offset = np.mean(analytic_signal_b)
    shifted_analytic_signal_b = analytic_signal_b - offset
    protophase = np.unwrap(np.angle(hilbert(signl_filtered) - offset))
    protophase_b = protophase[: stim_start]
    mapping = map_protophasse_to_phase(protophase_b, n_bins=200, transient_inds=100, order=25)
    phase = mapping(protophase)
    return phase

def get_intrinsic_frequency(phase, dt):
    omega, c = minimize(line, x0=np.random.rand(2), args=(np.arange(len(phase)) * dt, phase)).x
    return omega

def alpha_function(x, tau, x_0):
    res = ((x)/tau) * np.exp(-(x - x_0)/tau)
    return res / (sum(res))

def prepare_PRC_data(dataset_chunks, window):
    # contain tuple  (d phi/ dt - omega, phi) and array of values of perturbation function
    prc_data = []
    list_chunks = list(dataset_chunks.keys())
    for chunk_num in tqdm(list_chunks):
        window_function = - np.ones(window) / window #np.exp(-np.arange(window)/150)/sum(np.exp(-np.arange(window)/150)) #
        data_chunk = dataset_chunks[chunk_num]
        PNA = data_chunk['PNA']
        dt = dataset_chunks[chunk_num]['dt']
        ind_stim_start = dataset_chunks[chunk_num]['stim_start']
        phase = get_phase(PNA, ind_stim_start)
        omega = get_intrinsic_frequency(phase[:ind_stim_start], dt)
        phi = phase[ind_stim_start : ind_stim_start + window]
        d_phi_dt_minus_w = (np.diff(phase)/dt - omega)[ind_stim_start : ind_stim_start + window]
        prc_data.append((phi, d_phi_dt_minus_w, window_function))
    return prc_data

def extract_PRC(prc_data, num_coeffs):
    # after all the tuples are collected:
    for i in range(len(prc_data)):
        # min sum((Gx - h)**2)
        # x - Fourier coefficients ((a, b))
        # G - columns are cos (n phi) and sin(n phi)
        # h - dot phi - omega / П(phi)
        phi, d_phi_dt_minus_w, window_function = prc_data[i]
        G_ = np.hstack([np.hstack([np.cos(j * phi).reshape(-1, 1) for j in range(num_coeffs)]),
                       np.hstack([np.sin(j * phi).reshape(-1, 1) for j in range(num_coeffs)])])
        h_ = (d_phi_dt_minus_w / window_function).reshape(-1, 1)
        if i == 0:
            G = deepcopy(G_)
            h = deepcopy(h_)
        else:
            G = np.vstack([G, G_])
            h = np.vstack([h, h_])
    coeffs = lstsq(G, h)[0]

    def Z(x, coeffs):
        res = np.zeros_like(x)
        for i in range(len(coeffs)//2):
            res += coeffs[i] * np.cos(i * x) + coeffs[i + len(coeffs)//2] * np.sin(i * x)
        return res

    Phi = np.linspace(0, 2 * np.pi, 500)
    Delta_Phi = Z(Phi, coeffs)
    data = np.hstack([np.array(Phi).reshape(-1, 1),np.array(Delta_Phi).reshape(-1, 1)])
    return data

if __name__ == '__main__':
    method = 'prc_regression'
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"

    data_folder = f'{data_path}/sln_prc_chunked'
    folders = get_folders(data_folder, "_prc")
    window = 500

    # Prepare data
    for i, folder in enumerate(folders):
        file = f'chunked.pkl'
        dataset_chunks = pickle.load(open(f'{data_folder}/{folder}/{file}', 'rb+'))
        prc_data = (prepare_PRC_data(dataset_chunks, window))
        save_to = f'{data_path}/exp_results_phase_shift/{method}/temp_data/{folder}'
        create_dir_if_not_exist(save_to)
        pickle.dump(prc_data, open(save_to + f"/data.pkl", 'wb+'))

    ind_datasets = [0, 1,2, 3]
    prc_data = []
    for i, folder in enumerate(folders):
        if i in ind_datasets:
            load_from = f'{data_path}/exp_results_phase_shift/{method}/temp_data/{folder}'
            prc_data.extend(pickle.load(open(load_from + f"/data.pkl", 'rb+')))

    data = np.array(extract_PRC(prc_data, num_coeffs=15))
    data = data[data[:, 0].argsort(), :]
    Phi = data[:, 0]
    Delta_Phi = data[:, 1]
    plt.plot(Phi, Delta_Phi)
    plt.grid(True)
    plt.show(block=True)

