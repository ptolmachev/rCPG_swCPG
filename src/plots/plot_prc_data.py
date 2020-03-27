import pickle
import numpy as np
from utils.plot_utils import scatterplot


def features_plots(file_load, dir_save_to, modifier, fit_poly):
    data_ = pickle.load(open(file_load,'rb'))['data']
    data_ = np.array(data_)
    # data = get_rid_of_outliers(data)
    Phi = data_[:, 0]
    Ti_0 = data_[:, 1]
    T0 = data_[:, 2]
    T1 = data_[:, 3]
    Theta = data_[:, 4]
    Ti_1 = data_[:, 5]
    Ti_2 = data_[:, 6]
    phase = (Phi/np.nanmean(T0))
    cophase = (Theta/np.nanmean(T0))
    phi_insp = np.nanmean(Ti_0)/np.nanmean(T0)

    x_label = "Phase"
    x = phase
    x_lim = [0, 1.2]

    title = "Phase-Cophase"
    y_label = "Cophase"
    y = cophase
    y_lim = [0, 1.1]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=6)

    title = "T1 div T0"
    y_label = "T1/T0"
    y = T1/T0
    y_lim = [0, 1.75]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=6)

    title = "T0"
    y_label = "T0, ms"
    y = T0
    y_lim = [0, 5000]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=0)

    title = "T1"
    y_label = "T1, ms"
    y = T1
    y_lim = [0,5000]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim, title, fit_poly, path_save_to, phi_insp=phi_insp, deg=6)

    title = "Ti 1"
    y_label = "Ti 1, ms"
    y = Ti_1
    y_lim = [0,1500]
    path_save_to = f"{dir_save_to}/{title}_{modifier}.png"
    scatterplot(x, y, x_label, y_label, x_lim, y_lim , title, fit_poly, path_save_to, phi_insp=phi_insp, deg=0)
    return None



for i in range(4):
    num_rec = i
    data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                  '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
    file_load = f'../data/parameters_prc_{data_files[num_rec]}.pkl'
    dir_save_to = '../img/experiments'
    modifier = num_rec
    features_plots(file_load, dir_save_to, modifier, fit_poly=True)