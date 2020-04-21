import pickle

from matplotlib import  pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from utils.gen_utils import get_project_root, get_folders
from prc_extraction_regression import extract_PRC
from numpy.polynomial.polynomial import Polynomial

data_path = str(get_project_root()) + "/data"
img_path = str(get_project_root()) + "/img"
methods = ['prc_linear_fit', 'prc_threshold', 'prc_regression']
colors = ['red', 'blue', 'forestgreen']
ind_datasets = [0, 1, 2, 3]
fig = plt.figure(figsize=(20,10))

for i, method in enumerate(methods):
    if i != 2:
        data_folder = f'{data_path}/sln_prc_chunked'
        folders = get_folders(data_folder, "_prc")
        data = []
        data_folder = f'{data_path}/exp_results_phase_shift/{method}/temp_data/'
        folders = get_folders(data_folder, "_prc")
        for j, folder in enumerate(folders):
            if j in ind_datasets:
                load_from = f'{data_path}/exp_results_phase_shift/{method}/temp_data/{folder}'
                data.append(pickle.load(open(load_from + f"/data.pkl", 'rb+')))
        data = np.vstack(data)
        data = data[data[:, 0].argsort(), :]
        Phi = data[:, 0]
        Delta_Phi = data[:, 1]

        plt.scatter(Phi, Delta_Phi)#, color = colors[i])

        if i == 0:
            poly = Polynomial.fit(Phi, Delta_Phi, deg=10)
            plt.plot(Phi, poly(Phi), color='r')
            plt.scatter(Phi, savgol_filter(Delta_Phi,11,3), color='k')
    else:
        ind_datasets = [0, 1, 2, 3]
        prc_data = []
        for i, folder in enumerate(folders):
            if i in ind_datasets:
                load_from = f'{data_path}/exp_results_phase_shift/{method}/temp_data/{folder}'
                prc_data.extend(pickle.load(open(load_from + f"/data.pkl", 'rb+')))

        data = np.array(extract_PRC(prc_data, num_coeffs=15))
        data = data[data[:, 0].argsort(), :]
        Phi = data[:, 0]
        Delta_Phi = data[:, 1]
        plt.plot(Phi, 4 * Delta_Phi)

plt.grid(True)
plt.xlim([0, 2*np.pi])
plt.ylim([-2*np.pi, 2*np.pi])
plt.show(block=True)

fig2 = plt.figure(figsize = (20,10))
plt.plot()