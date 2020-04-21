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

def load_experimental_data(inds):
    #inds of datasets (which ones to use)
    img_path = str(get_project_root()) + "/img"
    data_path = str(get_project_root()) + "/data"
    data_files = ['2019-09-03_15-01-54_prc', '2019-09-04_17-49-02_prc',
                  '2019-09-05_12-26-14_prc', '2019-08-22_16-18-36_prc']
    phase = np.array([])
    cophase = np.array([])
    Phase_shift = np.array([])
    T0 = np.array([])
    T1 = np.array([])
    for i in inds:
        num_rec = i
        file_load = data_path + '/' + f'/sln_prc_params/parameters_prc_{data_files[num_rec]}.pkl'
        data_ = pickle.load(open(file_load, 'rb'))['data']
        data_ = np.array(data_)
        Phi_ = data_[:, 0].squeeze()
        T0_ = data_[:, 2].squeeze()
        T1_ = data_[:, 3].squeeze()
        Theta_ = data_[:, 4].squeeze()
        Delta_Phi_ = data_[:, 6].squeeze()
        phase = np.hstack([phase, (Phi_ / T0_)])
        cophase = np.hstack([cophase, (Theta_ / T0_)])
        T0 = np.hstack([T0, T0_])
        T1 = np.hstack([T1, T1_])
        Phase_shift = np.hstack([Phase_shift, (Delta_Phi_/T0_)])
    return phase, cophase, Phase_shift, T0, T1

def load_numerical_data(file_load):
    data_ = pickle.load(open(file_load,'rb'))['data']
    data_ = np.array(data_)
    # data = get_rid_of_outliers(data)
    Phi = data_[:, 0]
    Ti_0 = data_[:, 1]
    T0 = data_[:, 2]
    T1 = data_[:, 3]
    Theta = data_[:, 4]
    Ti_1 = data_[:, 5]
    # Delta_Phi = data_[:, 6]
    Phase = (Phi/(T0))
    Cophase = (Theta/(T0))
    # Phase_shift = (Delta_Phi / (T0))
    return Phase, Cophase, T0, T1 #Phase_shift

def plot_comparisons(amp, duration, plot_params):
    img_path = str(get_project_root()) + "/img"
    data_path = str(get_project_root()) + "/data"
    dir_save_to = img_path + '/' + "experiments"
    fit_poly = plot_params["fit_poly"]
    y_lim_T0 = plot_params["y_lim_T0"]
    y_lim_T1 = plot_params["y_lim_T1"]
    y_lim_Ti1 = plot_params["y_lim_Ti1"]
    y_lim_T1divT0 = plot_params["y_lim_T1divT0"]

    #LOADING EXPERIMENTAL DATA
    inds = [0,1, 2, 3]
    Phase_exp, Cophase_exp, Phase_shift_exp, T0_exp, T1_exp = load_experimental_data(inds)

    # # LOADING NUMERICAL DATA
    # file_load = f'{data_path}/num_exp_results/short_stim/info_var_phase_{amp}_{duration}.pkl'
    # # Phase_num, Cophase_num, Phase_shift_num, T0_num, T1_num = load_numerical_data(file_load)
    # Phase_num, Cophase_num, T0_num, T1_num = load_numerical_data(file_load)
    # fig1 = plt.figure()
    # plt.title("Phase-Cophase")
    # y = Cophase_exp
    # z = Cophase_num
    # p = Polynomial.fit(Phase_exp, y, deg=6)
    # plt.scatter(Phase_exp, y)
    # plt.scatter(Phase_num, z, color = "orange")
    # plt.plot(np.sort(Phase_exp), p(np.sort(Phase_exp)), color='r', linewidth=3)
    # plt.grid(True)
    # plt.xlabel("Phase")
    # plt.ylabel("Cophase")
    # fig1.savefig(f'{img_path}/num_experiments/short_stim/short_stim_{amp}_{duration}/phase_cophase_comparison')
    # # #
    # fig2 = plt.figure()
    # plt.title("T1/T0")
    # y = T1_exp / T0_exp
    # z = T1_num/ T0_num
    # p = Polynomial.fit(Phase_exp, y, deg=6)
    # plt.scatter(Phase_exp, y)
    # plt.scatter(Phase_num, z, color = "orange")
    # plt.plot(np.sort(Phase_exp), p(np.sort(Phase_exp)), color='r', linewidth=3)
    # plt.grid(True)
    # # plt.ylim([0, 1.1 * np.max(y - np.mean(y)) + np.mean(y)])
    # plt.ylim([0, 2])
    # plt.xlabel("Phase")
    # plt.ylabel("T1/T0 ratio")
    # fig2.savefig(f'{img_path}/num_experiments/short_stim/short_stim_{amp}_{duration}/T1_div_T0_comparison')
    #
    fig3 = plt.figure(figsize=(20,20))
    plt.title("delta Phi(Phi)", fontsize=24)
    y = Phase_shift_exp
    z = (Phase_exp + Cophase_exp - 1) # phase shift from different perspective
    Phase_exp, y, z = zip(*sorted(zip(Phase_exp, y, z)))

    from scipy.signal import savgol_filter
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = (np.hstack([np.array(Phase_exp).reshape(-1,1), np.array(y).reshape(-1,1)]))
    X = scaler.fit_transform(X)
    clf = BayesianGaussianMixture(n_components=2, covariance_type='full', init_params='random')
    mask = clf.fit_predict(X)
    inds_1 = np.where(mask == 0)[0]
    inds_2 = np.where(mask == 1)[0]
    Phase_exp_1 = np.take(Phase_exp, inds_1)
    Phase_exp_2 = np.take(Phase_exp, inds_2)
    y_1 = np.take(y, inds_1)
    y_2 = np.take(y, inds_2)
    plt.scatter(Phase_exp_1[10:], savgol_filter(y_1[10:],3,1), color='orange')
    plt.scatter(Phase_exp_2[10:], savgol_filter(y_2[10:],3,1), color='magenta')
    # plt.scatter(Phase_exp, y, color = 'b', label='Phase response advanced')
    plt.scatter(Phase_exp, z, color = 'r', marker='x', alpha = 0.5, label='Phase response threshold')
    # plt.scatter(phase_num, z, color = "orange")
    # plt.plot(np.sort(phase), p(np.sort(phase)), color='r', linewidth=3)
    plt.grid(True)
    plt.ylim([-1, 1])
    plt.xlabel("Phase",fontsize=24)
    plt.ylabel("delta Phi",fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.show(block=True)
    fig3.savefig(f'{img_path}/other_plots/delta_Phi_0')

    return None

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
    plot_comparisons(amp, duration, plot_params)