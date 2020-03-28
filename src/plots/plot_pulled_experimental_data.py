import pickle
from numpy.polynomial.polynomial import Polynomial
from utils.gen_utils import get_project_root
import numpy as np
from matplotlib import pyplot as plt

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
    plt.ylim([0, 1.1 * np.max(y - np.mean(y)) + np.mean(y)])
    plt.xlabel("Phase")
    plt.ylabel("T1/T0 ratio")
    plt.savefig(f"{dir_save_to}/pulled_period_change.png")
    return None


if __name__ == '__main__':
    combining_data(inds=[0, 1, 2]) # '2019-08-22_16-18-36_prc' - outlier
