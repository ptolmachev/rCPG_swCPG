import pickle
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from utils.gen_utils import get_project_root
from utils.plot_utils import features_plots

def combining_data(inds):
    img_path = str(get_project_root()) + "/img"
    data_path = str(get_project_root()) + "/data"
    data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                  '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
    phase = np.array([])
    cophase = np.array([])
    T0_all = np.array([])
    T1_all = np.array([])
    dir_save_to = img_path + '/' + "experiments"
    for i in inds:
        num_rec = i
        file_load = data_path + '/' + f'/sln_prc_params/parameters_prc_{data_files[num_rec]}.pkl'
        data_ = pickle.load(open(file_load, 'rb'))['data']
        data_ = np.array(data_)
        Phi = data_[:, 0]
        T0 = data_[:, 2]
        T1 = data_[:, 3]
        Theta = data_[:, 4]
        phase = np.hstack([phase, (Phi / np.mean(T0))])
        cophase = np.hstack([cophase, (Theta / np.mean(T0))])
        T0_all = np.hstack([T0_all, T0])
        T1_all = np.hstack([T1_all, T1])

    fig1 = plt.figure()
    plt.title("Phase-Cophase")
    y = cophase
    p = Polynomial.fit(phase, y,deg=6)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.xlabel("Phase")
    plt.ylabel("Cophase")
    plt.savefig(f"{dir_save_to}/pulled_phase_cophase.png")

    fig2 = plt.figure()
    plt.title("T1/T0")
    y = T1_all/T0_all
    p = Polynomial.fit(phase, y,deg=6)
    plt.scatter(phase, y)
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    # plt.ylim([0, 1.1 * np.max(y - np.mean(y)) + np.mean(y)])
    plt.ylim([0, 2])
    plt.xlabel("Phase")
    plt.ylabel("T1/T0 ratio")
    plt.savefig(f"{dir_save_to}/pulled_period_change.png")
    return None

def plot_feature_from_experiment(plot_params):
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"
    for i in range(4):
        num_rec = i
        data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                      '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
        file_load = f'{data_path}/sln_prc_params/parameters_prc_{data_files[num_rec]}.pkl'
        dir_save_to = f'{img_path}/experiments'
        modifier = num_rec
        features_plots(file_load, dir_save_to, modifier, plot_params)
    return None

def plot_feature_from_num_experiment(amp, duration, plot_params):
    data_path = str(get_project_root()) + "/data"
    img_path = str(get_project_root()) + "/img"
    file_load = f'{data_path}/num_exp_results/short_stim/info_var_phase_{amp}_{duration}.pkl'
    dir_save_to = f'{img_path}/num_experiments/short_stim/short_stim_{amp}_{duration}'
    modifier = f"{amp}_{duration}"
    features_plots(file_load, dir_save_to, modifier, plot_params)
    return None

def plot_comparison(amp, duration, plot_params):
    img_path = str(get_project_root()) + "/img"
    data_path = str(get_project_root()) + "/data"

    #LOADING EXPERIMENTAL DATA
    data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                  '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
    inds = [0, 1, 2]
    phase = np.array([])
    cophase = np.array([])
    T0_all = np.array([])
    T1_all = np.array([])
    dir_save_to = img_path + '/' + "experiments"
    for i in inds:
        num_rec = i
        file_load = data_path + '/' + f'/sln_prc_params/parameters_prc_{data_files[num_rec]}.pkl'
        data_ = pickle.load(open(file_load, 'rb'))['data']
        data_ = np.array(data_)
        Phi = data_[:, 0]
        T0 = data_[:, 2]
        T1 = data_[:, 3]
        Theta = data_[:, 4]
        phase = np.hstack([phase, (Phi / np.mean(T0))])
        cophase = np.hstack([cophase, (Theta / np.mean(T0))])
        T0_all = np.hstack([T0_all, T0])
        T1_all = np.hstack([T1_all, T1])

    fit_poly = plot_params["fit_poly"]
    y_lim_T0 = plot_params["y_lim_T0"]
    y_lim_T1 = plot_params["y_lim_T1"]
    y_lim_Ti1 = plot_params["y_lim_Ti1"]
    y_lim_T1divT0 = plot_params["y_lim_T1divT0"]

    #LOADING NUMERICAL DATA
    file_load = f'{data_path}/num_exp_results/short_stim/info_var_phase_{amp}_{duration}.pkl'
    data_ = pickle.load(open(file_load,'rb'))['data']
    data_ = np.array(data_)
    # data = get_rid_of_outliers(data)
    Phi_num = data_[:, 0]
    Ti_0_num = data_[:, 1]
    T0_num = data_[:, 2]
    T1_num = data_[:, 3]
    Theta_num = data_[:, 4]
    Ti_1_num = data_[:, 5]
    Ti_2_num = data_[:, 6]
    phase_num = (Phi_num/np.nanmean(T0_num))
    cophase_num = (Theta_num/np.nanmean(T0_num))
    phi_insp_num = np.nanmean(Ti_0_num)/np.nanmean(T0_num)

    fig1 = plt.figure()
    plt.title("Phase-Cophase")
    y = cophase
    z = cophase_num
    p = Polynomial.fit(phase, y, deg=6)
    plt.scatter(phase, y)
    plt.scatter(phase_num, z, color = "orange")
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.xlabel("Phase")
    plt.ylabel("Cophase")
    fig1.savefig(f'{img_path}/num_experiments/short_stim/short_stim_{amp}_{duration}/phase_cophase_comparison')

    fig2 = plt.figure()
    plt.title("T1/T0")
    y = T1_all / T0_all
    z = T1_num/ T0_num
    p = Polynomial.fit(phase, y, deg=6)
    plt.scatter(phase, y)
    plt.scatter(phase_num, z, color = "orange")
    plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    # plt.ylim([0, 1.1 * np.max(y - np.mean(y)) + np.mean(y)])
    plt.ylim([0, 2])
    plt.xlabel("Phase")
    plt.ylabel("T1/T0 ratio")
    fig2.savefig(f'{img_path}/num_experiments/short_stim/short_stim_{amp}_{duration}/T1_div_T0_comparison')

    #get pulled experimental data

if __name__ == '__main__':
    # EXPERIMENTAL PRC
    # plot_params = {}
    # plot_params["fit_poly"] = True
    # plot_params["y_lim_T0"] = [0, 5000]
    # plot_params["y_lim_T1"]  = [0, 5000]
    # plot_params["y_lim_Ti1"] = [0, 1500]
    # plot_params["y_lim_T1divT0"] = [0, 2]
    # plot_feature_from_experiment(plot_params)
    # inds = [0, 1, 2]
    # combining_data(inds)


    #NUMERICAL PRC
    # amp = 250
    # duration = 500
    # plot_params = {}
    # plot_params["fit_poly"] = True
    # plot_params["y_lim_T0"] = [0, 7500]
    # plot_params["y_lim_T1"] = [0, 7500]
    # plot_params["y_lim_Ti1"] = [0, 2500]
    # plot_params["y_lim_T1divT0"] = [0, 2]
    # plot_feature_from_num_experiment(amp, duration, plot_params)


    amp = 150
    duration = 500
    plot_params = {}
    plot_params["fit_poly"] = True
    plot_params["y_lim_T0"] = [0, 7500]
    plot_params["y_lim_T1"] = [0, 7500]
    plot_params["y_lim_Ti1"] = [0, 2500]
    plot_params["y_lim_T1divT0"] = [0, 2]
    plot_comparison(amp, duration, plot_params)